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

/*! \file samples/closestPointOnTrianglesSurface Simple example of
    building bvhes over, and quering closest points on, sets of 3D
    triangles

    This example will, in successive steps:

    1) load a cmdline-specified OBJ file of triangles

    2) build BVH over those triangles

    3) run some sample find-closst-point queries: generate a grid of
    512x512x512 cells (stretched over the bounding box of the model),
    then for each cell center, perform a bvh fcp closest-point query
    on those line segmetns.
*/

// cuBQL:
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/triangles/fcp.h"
#include "testing/helper/triangles.h"

// std:
#include <random>
#include <fstream>

using cuBQL::vec3i;
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

/*! generate boxes (required for bvh builder) from prim type 'index line triangles' */
__global__ void generateBoxes(box3f *boxForBuilder,
                              const vec3i *triangles,
                              int numTriangles,
                              const vec3f *vertices)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numTriangles) return;
  
  auto triangle = triangles[tid];
  box3f bbox
    = box3f(/*empty box*/)
    .including(vertices[triangle.x])
    .including(vertices[triangle.y])
    .including(vertices[triangle.z]);
  boxForBuilder[tid] = bbox;
}


/*! the actual sample query: generates points in a gridDim^3 grid of points, then for each such grid point perform a query */
__global__
void runQueries(float       *results,
                int          gridDim,
                bvh3f        trianglesBVH,
                const vec3i *triangles,
                const vec3f *vertices,
                bool         firstTime)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int numGridPoints = gridDim * gridDim * gridDim;
  if (tid >= numGridPoints) return;

  int ix = tid % gridDim;
  int iy = (tid / gridDim) % gridDim;
  int iz = tid / (gridDim*gridDim);

  // hack: this _must_ be the bounding box of the triangles:
  const box3f bbox = trianglesBVH.nodes[0].bounds;
  
  vec3f relQueryPoint = vec3f{float((ix+.5f)/gridDim),
                              float((iy+.5f)/gridDim),
                              float((iz+.5f)/gridDim)};
  vec3f queryPoint = bbox.lower + relQueryPoint * bbox.size();
  cuBQL::triangles::FCPResult result;
  result.clear(INFINITY);
  cuBQL::triangles::fcp(result,queryPoint,
                        trianglesBVH,triangles,vertices);
  results[tid] = 10.f*sqrtf(result.sqrDistance);
  
  if (firstTime && ((tid % 10000000) == 13))
    printf("for reference: closest triangle to point (%f %f %f) is triangle %i, connecting %i:(%f %f %f), %i:(%f %f %f), and  %i:(%f %f %f); at distance %f\n",
           queryPoint.x,
           queryPoint.y,
           queryPoint.z,
           result.primID,
           triangles[result.primID].x,
           vertices[triangles[result.primID].x].x,
           vertices[triangles[result.primID].x].y,
           vertices[triangles[result.primID].x].z,
           triangles[result.primID].y,
           vertices[triangles[result.primID].y].x,
           vertices[triangles[result.primID].y].y,
           vertices[triangles[result.primID].y].z,
           triangles[result.primID].z,
           vertices[triangles[result.primID].z].x,
           vertices[triangles[result.primID].z].y,
           vertices[triangles[result.primID].z].z,
           result.sqrDistance);
}


int main(int ac, const char **av)
{
  const char *inFileName = "../samples/bunny.obj";
  if (ac != 1)
    inFileName = av[1];
  
  // ------------------------------------------------------------------
  // step 1: load triangle mesh
  // ------------------------------------------------------------------
  std::vector<vec3i> h_indices;
  std::vector<vec3f> h_vertices;
  std::cout << "loading triangles from " << inFileName << std::endl;
  cuBQL::test_rig::loadOBJ(h_indices,h_vertices,inFileName);
  std::cout << "loaded OBJ file, got " << prettyNumber(h_indices.size())
            << " triangles with " << prettyNumber(h_vertices.size())
            << " vertices" << std::endl;
  
  // upload to the device:
  vec3f *vertices = allocManaged<vec3f>((int)h_vertices.size());
  std::copy(h_vertices.begin(),h_vertices.end(),vertices);
  vec3i *indices  = allocManaged<vec3i>((int)h_indices.size());
  std::copy(h_indices.begin(),h_indices.end(),indices);
  int numTriangles = (int)h_indices.size();
  
  // ------------------------------------------------------------------
  // step 2) build BVH over those triangles, so we can do queries on
  // them
  // ------------------------------------------------------------------
  
  bvh3f trianglesBVH;
  {
    
    // allocate memory for bounding boxes (to build BVH over)
    box3f *boxes = allocManaged<box3f>(numTriangles);
    
    // run cuda kernel that generates a bounding box for each point 
    generateBoxes<<<divRoundUp(numTriangles,1024),1024>>>
      (boxes,indices,numTriangles,vertices);
    
    // ... aaaand build the BVH
    cuBQL::BuildConfig buildConfig = /*default:*/{};
    double t0 = getCurrentTime();
    cuBQL::gpuBuilder(trianglesBVH,boxes,numTriangles,buildConfig);
    double t1 = getCurrentTime();
    // free the boxes - we could actually re-use that memory below, but
    // let's just do this cleanly here.
    cudaFree(boxes);
    std::cout << "done building BVH over " << prettyNumber(numTriangles)
              << " triangles, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  // ------------------------------------------------------------------
  // step 3: run some sample query:
  // ------------------------------------------------------------------
  
  int gridDim = 512;
  int numQueries = gridDim * gridDim * gridDim;
  
  // allocate memory for results:
  float *sqrDist = allocManaged<float>(numQueries);
  runQueries<<<divRoundUp(numQueries,1024),1024>>>
    (sqrDist,gridDim,trianglesBVH,indices,vertices,true);
  CUBQL_CUDA_SYNC_CHECK();
  
  std::cout << "for timing, running it again:" << std::endl;
  double t0 = getCurrentTime();
  runQueries<<<divRoundUp(numQueries,1024),1024>>>
    (sqrDist,gridDim,trianglesBVH,indices,vertices,false);
  CUBQL_CUDA_SYNC_CHECK();
  double t1 = getCurrentTime();
  std::cout << " .... took " << prettyDouble(t1-t0) << "s for those "
            << gridDim << "x" << gridDim << "x" << gridDim
            << " (= " << prettyNumber(gridDim*gridDim*gridDim) << ")"
            << " queries..." << std::endl;

#if 0
  // saving to raw volume file, can look at that for sanity testing
  std::cout << "saving to distances.raw" << std::endl;
  std::ofstream out("distances.raw",std::ios::binary);
  out.write((const char *)sqrDist,numQueries*sizeof(float));
#endif
  return 0;
}
