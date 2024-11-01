// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/*! \file samples/lineSegs3f Simple example of building bvhes over,
    and querying, 3D line segments.

    This example will, in successive steps:

    1) generate 1M random points as seed points (to use for generating line
    segments)

    2) build BVH over those seed points, so we can run fcp queries to
    find closest-pairs (to use as line segments)

    3) compute all-closest-pairs between these points, using the
    _point_ fcp kernel; and create a line segment (LineSeg3f) for each
    closest-pairs of poins

    4) build bvh over those line segments

    5) run some sample find-closst-point-on-line-segments query:
    generate a grid of 512x512x512 cells, for each cell center,
    perform a bvh fcp closest-point query on those line segments.
*/

// cuBQL:
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/lineSegs/LineSegs3f.h"
#include "cuBQL/points/fcp.h"

// std:
#include <random>

using cuBQL::vec3f;
using cuBQL::box3f;
using cuBQL::bvh3f;
using cuBQL::divRoundUp;
using cuBQL::prettyNumber;
using cuBQL::prettyDouble;
using cuBQL::getCurrentTime;

/*! helper function that allocates managed memory, and cheks for errors */
template<typename T>
T *allocManaged(int N)
{
  T *ptr = 0;
  CUBQL_CUDA_CALL(MallocManaged((void **)&ptr,N*sizeof(T)));
  return ptr;
}

/*! generate boxes (required for bvh builder) from prim type 'point' */
__global__ void generateBoxes(box3f *boxesForBuilder,
                              const vec3f *inputPoints,
                              int numPoints)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numPoints) return;
  
  const vec3f point = inputPoints[tid];
  boxesForBuilder[tid] = box3f{ point,point };
}

/*! generate boxes (required for bvh builder) from prim type 'index line segments' */
__global__ void generateBoxes(box3f *boxForBuilder,
                              const cuBQL::lineSegs::IndexedSegment *segments,
                              int numSegments,
                              const vec3f *vertices)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numSegments) return;
  
  auto segment = segments[tid];
  box3f segmentBounds
    = box3f(/*empty box*/)
    .including(vertices[segment.begin])
    .including(vertices[segment.end]);
  boxForBuilder[tid] = segmentBounds;
}


/*! helper routine that generates line segments by finding all closest
    pairs in a populoation of input points. this also serves as a demo
    of using point-to-point fcp queries, but the main purose is to
    generate useful line segments */
__global__
void createClosestPairSegments(int *pNumSegmentsFound,
                               cuBQL::lineSegs::IndexedSegment *foundSegments,
                               /*! computed bvh over seed points, so
                                   we can do quries*/
                               bvh3f seedPointsBVH,
                               const vec3f *seedPoints,
                               int numSeedPoints)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numSeedPoints) return;

  vec3f sourcePoint = seedPoints[tid];
  // find closst point that is _not_ the point itself
  int partner
    = cuBQL::points::fcp_excluding(tid,
                                   sourcePoint,
                                   seedPointsBVH,seedPoints);
  // in most (but not all) cases, if a is fcp to b then b will be fcp
  // to a. we want to avoid reporting _both_ (a,b) _and_ (b,a), but
  // should discard one of them only if they willactually both be
  // reported
  int partner_of_partner
    = cuBQL::points::fcp_excluding(partner,
                                   seedPoints[partner],
                                   seedPointsBVH,seedPoints);
  if ((partner_of_partner == tid) && (partner > tid))
    // avoid duplicate:
    return;

  int newSegID = atomicAdd(pNumSegmentsFound,1);
  foundSegments[newSegID] = { tid,partner };
}


/*! the actual sample query: generates points in a gridDim^3 grid of points, then for each such grid point perform a query */
__global__
void runQueries(float *results,
                int gridDim,
                bvh3f segmentsBVH,
                const cuBQL::lineSegs::IndexedSegment *segments,
                const vec3f                           *vertices,
                bool firstTime)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int numGridPoints = gridDim * gridDim * gridDim;
  if (tid >= numGridPoints) return;

  int ix = tid % gridDim;
  int iy = (tid / gridDim) % gridDim;
  int iz = tid / (gridDim*gridDim);
  
  vec3f queryPoint = vec3f{float((ix+.5f)/gridDim),
                           float((iy+.5f)/gridDim),
                           float((iz+.5f)/gridDim)};
  cuBQL::lineSegs::FCPResult result;
  result.clear(INFINITY);
  cuBQL::lineSegs::fcp(result,queryPoint,
                       segmentsBVH,segments,vertices);
  results[tid] = result.sqrDistance;
  
  if (firstTime && ((tid % 10000000) == 13))
    printf("for reference: closest segment to point (%f %f %f) is seg %i, connecting %i:(%f %f %f) and %i:(%f %f %f), at distance %f\n",
           queryPoint.x,
           queryPoint.y,
           queryPoint.z,
           result.primID,
           segments[result.primID].begin,
           vertices[segments[result.primID].begin].x,
           vertices[segments[result.primID].begin].y,
           vertices[segments[result.primID].begin].z,
           segments[result.primID].end,
           vertices[segments[result.primID].end].x,
           vertices[segments[result.primID].end].y,
           vertices[segments[result.primID].end].z,
           result.sqrDistance);
}


int main(int ac, char **av)
{
  int numSeedPoints = 1000000;

  // ------------------------------------------------------------------
  // step 1: generate random seed points
  // ------------------------------------------------------------------

  // alloc memory to store the seed points, accessible on both dev and
  // host
  vec3f *seedPoints = allocManaged<vec3f>(numSeedPoints);

  // generate some uniform random data...
  std::default_random_engine rng;
  rng.seed(0x12345);
  std::uniform_real_distribution<float> uniform(0.f,10.f);
  for (int i=0;i<numSeedPoints;i++)
    seedPoints[i] = vec3f{ uniform(rng),uniform(rng),uniform(rng) };
  std::cout << "generated " << numSeedPoints << " seed points..." << std::endl;
  CUBQL_CUDA_SYNC_CHECK();
  // ------------------------------------------------------------------
  // step 2) build BVH over those seed points, so we can run fcp
  // queries to find cloest-pairs (to use asline segments)
  // ------------------------------------------------------------------
  
  bvh3f seedPointsBVH;
  {
    // allocate memory for bounding boxes (to build BVH over)
    box3f *boxes = allocManaged<box3f>(numSeedPoints);
    
    // run cuda kernel that generates a bounding box for each point 
    generateBoxes<<<divRoundUp(numSeedPoints,1024),1024>>>
      (boxes,seedPoints,numSeedPoints);
    CUBQL_CUDA_SYNC_CHECK();
    
    // ... aaaand build the BVH
    cuBQL::BuildConfig buildConfig = /*default:*/{};
    cuBQL::gpuBuilder(seedPointsBVH,boxes,numSeedPoints,buildConfig);
    CUBQL_CUDA_SYNC_CHECK();
    
    // free the boxes - we could actually re-use that memory below, but
    // let's just do this cleanly here.
    cudaFree(boxes);
    
    std::cout << "built bvh over seed points, got "
              << seedPointsBVH.numNodes << " nodes..." << std::endl;
  }
  
  
  // ------------------------------------------------------------------
  // step 3: bvh is built (means we can do fcp queries): generate all
  // pairs
  // ------------------------------------------------------------------

  // pre-alloc meomry for all _possible_ pairs
  int maxNumSegments = numSeedPoints;
  cuBQL::lineSegs::IndexedSegment *segments
    = allocManaged<cuBQL::lineSegs::IndexedSegment>(maxNumSegments);

  // allocate and init atomic counter to store _actual_ num segments
  // generated
  int *pNumSegments = allocManaged<int>(1);
  *pNumSegments = 0;
  createClosestPairSegments<<<divRoundUp(numSeedPoints,128),128>>>
    (pNumSegments,segments,seedPointsBVH,seedPoints,numSeedPoints);
  CUBQL_CUDA_SYNC_CHECK();
  int numSegments = *pNumSegments;
  
  std::cout << "done finding closest-segments, found " << numSegments << std::endl;
  if (numSegments == 0)
    throw std::runtime_error("error - could not find _any_ segments!?");
  
  // wo no longer need the seen points bvh, now - let's free
  cuBQL::free(seedPointsBVH);
  
  // ------------------------------------------------------------------
  // step 3: build BVH over segments
  // ------------------------------------------------------------------
  bvh3f segmentsBVH;
  {
    
    // allocate memory for bounding boxes (to build BVH over)
    box3f *boxes = allocManaged<box3f>(numSegments);
    
    // run cuda kernel that generates a bounding box for each point 
    generateBoxes<<<divRoundUp(numSegments,1024),1024>>>
      (boxes,segments,numSegments,seedPoints);
    
    // ... aaaand build the BVH
    cuBQL::BuildConfig buildConfig = /*default:*/{};
    double t0 = getCurrentTime();
    cuBQL::gpuBuilder(segmentsBVH,boxes,numSegments,buildConfig);
    double t1 = getCurrentTime();
    // free the boxes - we could actually re-use that memory below, but
    // let's just do this cleanly here.
    cudaFree(boxes);
    std::cout << "done building BVH over " << prettyNumber(numSegments)
              << " line segments, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  // ------------------------------------------------------------------
  // step 4: run some sample query:
  // ------------------------------------------------------------------
  
  int gridDim = 512;
  int numQueries = gridDim * gridDim * gridDim;
  
  // allocate memory for results:
  float *sqrDist = allocManaged<float>(numQueries);
  runQueries<<<divRoundUp(numQueries,1024),1024>>>
    (sqrDist,gridDim,segmentsBVH,segments,seedPoints,true);
  CUBQL_CUDA_SYNC_CHECK();
  
  std::cout << "for timing, running it again:" << std::endl;
  double t0 = getCurrentTime();
  runQueries<<<divRoundUp(numQueries,1024),1024>>>
    (sqrDist,gridDim,segmentsBVH,segments,seedPoints,false);
  CUBQL_CUDA_SYNC_CHECK();
  double t1 = getCurrentTime();
  std::cout << " .... took " << prettyDouble(t1-t0) << "s for those "
            << gridDim << "x" << gridDim << "x" << gridDim
            << " (= " << prettyNumber(gridDim*gridDim*gridDim) << ")"
            << " queries..." << std::endl;
  return 0;
}
