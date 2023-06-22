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

// #define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/computeSAH.h"
#include "cuBQL/queries/fcp.h"

#include "testing/helper/CUDAArray.h"
#include "testing/helper.h"
#include "testing/helper/Generator.h"

namespace testing {

  std::vector<float> reference;
  
  struct TestConfig {
    float maxTimeThreshold = 10.f;
    float maxQueryRadius = INFINITY;

    std::string dataDist = "uniform";
    int dataCount = 100000;
    std::string queryDist = "uniform";
    int queryCount = 100000;

    bool make_reference = false;
    std::string referenceFileName;
  };
  
  void usage(const std::string &error = "")
  {
    if (!error.empty()) {
      std::cerr << error << "\n\n";
    }
    std::cout << "./cuBQL_fcp dataPoints.dat queryPoints.dat\n\n";
    exit(error.empty()?0:1);
  }

  template<int D>
  __global__
  void makeBoxes(cuBQL::box_t<float,D> *boxes,
                 cuBQL::vec_t<float,D> *points,
                 int numPoints)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;
    vec_t<float,D> point = points[tid];
    boxes[tid].lower = point;
    boxes[tid].upper = point;
  }

  __global__
  void resetResults(float *results, int numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;
    results[tid] = INFINITY;//-1;
  }


  template<int D>
  inline __device__
  int fcp(const BinaryBVH<float,D> bvh,
          const vec_t<float,D>    *dataPoints,
          const vec_t<float,D>     query,
          /* in: SQUARE of max search distance; out: sqrDist of closest point */
          float          &maxQueryDistSquare)
  {
    int result = -1;
    
    int2 stackBase[32], *stackPtr = stackBase;
    int nodeID = 0;
    int offset = 0;
    int count  = 0;
    while (true) {
      while (true) {
        offset = bvh.nodes[nodeID].offset;
        count  = bvh.nodes[nodeID].count;
        if (count>0)
          // leaf
          break;
        typename BinaryBVH<float,D>::Node child0 = bvh.nodes[offset+0];
        typename BinaryBVH<float,D>::Node child1 = bvh.nodes[offset+1];
        float dist0 = fSqrDistance(child0.bounds,query);
        float dist1 = fSqrDistance(child1.bounds,query);
        int closeChild = offset + ((dist0 > dist1) ? 1 : 0);
        if (dist1 <= maxQueryDistSquare) {
          float dist = max(dist0,dist1);
          int distBits = __float_as_int(dist);
          *stackPtr++ = make_int2(closeChild^1,distBits);
        }
        if (min(dist0,dist1) > maxQueryDistSquare) {
          count = 0;
          break;
        }
        nodeID = closeChild;
      }
      for (int i=0;i<count;i++) {
        int primID = bvh.primIDs[offset+i];
        float dist2 = sqrDistance(dataPoints[primID],query);
        if (dist2 >= maxQueryDistSquare) continue;
        maxQueryDistSquare = dist2;
        result             = primID;
      }
      while (true) {
        if (stackPtr == stackBase)
          return result;
        --stackPtr;
        if (__int_as_float(stackPtr->y) > maxQueryDistSquare) continue;
        nodeID = stackPtr->x;
        break;
      }
    }
  }

  
  template<int D, typename bvh_t>
  __global__
  void runFCP(float        *results,
              bvh_t         bvh,
              const vec_t<float,D> *dataPoints,
              const vec_t<float,D> *queries,
              float         maxRadius,
              int           numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;

    const vec_t<float,D> query = queries[tid];
    float sqrMaxQueryDist = maxRadius*maxRadius;
    int res = fcp(bvh,dataPoints,query,
                  /* fcp kernel expects SQUARE of radius */
                  sqrMaxQueryDist);
    if (res == -1)
      results[tid] = INFINITY;
    else
      // results[tid] = sqrDistance(dataPoints[res],query);
      results[tid] = sqrMaxQueryDist;
  }
  
  template<int D, typename bvh_t=cuBQL::BinaryBVH<float,D>>
  void testFCP(TestConfig testConfig,
              BuildConfig buildConfig)
  {
    using point_t = cuBQL::vec_t<float,D>;
    using box_t = cuBQL::box_t<float,D>;
    typename PointGenerator<float,D>::SP dataGenerator
      = PointGenerator<float,D>::parse(testConfig.dataDist);
    typename PointGenerator<float,D>::SP queryGenerator
      = PointGenerator<float,D>::parse(testConfig.queryDist);
    // std::vector<point_t> h_dataPoints;
    // std::vector<point_t> h_queryPoints;
    
    CUDAArray<point_t> dataPoints;
    dataGenerator->generate(dataPoints,testConfig.dataCount);
    CUDAArray<point_t> queryPoints;
    queryGenerator->generate(queryPoints,testConfig.queryCount);
    // dataPoints.upload(h_dataPoints);

    CUDAArray<box_t> boxes(dataPoints.size());
    {
      int bs = 256;
      int nb = divRoundUp((int)dataPoints.size(),bs);
      makeBoxes<<<nb,bs>>>(boxes.data(),dataPoints.data(),(int)dataPoints.size());
    };

    // cuBQL::BinaryBVH
    bvh_t bvh;
    cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),buildConfig);
    if (D == 3) 
      std::cout << "done build, sah cost is " << cuBQL::computeSAH(bvh) << std::endl;
    else
      std::cout << "done build..." << std::endl;

    // CUDAArray<float3> queryPoints;
    // queryPoints.upload(h_queryPoints);
    
    int numQueries = queryPoints.size();
    CUDAArray<float> closest(numQueries);

    // ------------------------------------------------------------------
    std::cout << "first query for warm-up and checking reference data (if provided)"
              << std::endl;
    // ------------------------------------------------------------------
    resetResults<<<divRoundUp(numQueries,128),128>>>(closest.data(),numQueries);
    runFCP<<<divRoundUp(numQueries,128),128>>>
      (closest.data(),
       bvh,
       dataPoints.data(),
       queryPoints.data(),
       testConfig.maxQueryRadius,
       numQueries);
    CUBQL_CUDA_SYNC_CHECK();
    if (reference.empty()) {
      reference = closest.download();
    } else {
      std::cout << "checking reference...." << std::endl;
      std::vector<float> ours = closest.download();
      if (ours.size() != reference.size())
        throw std::runtime_error("reference file size odes not have expected number of entries!?");
      for (int i=0;i<ours.size();i++) {
        if (ours[i] > reference[i] && sqr(reference[i]) <= sqr(testConfig.maxQueryRadius)) {
          std::cout << "ours/reference mismatch at index "
                    << i << "/" << reference.size() << ":" << std::endl;
          std::cout << "  ours      is " << ours[i] << std::endl;
          std::cout << "  reference is " << reference[i] << std::endl;
          throw std::runtime_error("does NOT match!");
        }
      }
      std::cout << "all good, ours matches reference array ..." << std::endl;
    }

    // ------------------------------------------------------------------
    // actual timing runs
    // ------------------------------------------------------------------
    int numPerRun = 1;
    while (true) {
      CUBQL_CUDA_SYNC_CHECK();
      std::cout << "timing run with " << numPerRun << " repetition.." << std::endl;
      double t0 = getCurrentTime();
      for (int i=0;i<numPerRun;i++) {
        resetResults<<<divRoundUp(numQueries,128),128>>>(closest.data(),numQueries);
        runFCP<<<divRoundUp(numQueries,128),128>>>
          (closest.data(),
           bvh,
           dataPoints.data(),
           queryPoints.data(),
           testConfig.maxQueryRadius,
           numQueries);
        CUBQL_CUDA_SYNC_CHECK();
      }
      double t1 = getCurrentTime();
      std::cout << "done " << numPerRun
                << " queries in " << prettyDouble(t1-t0) << "s, that's "
                << prettyDouble((t1-t0)/numPerRun) << "s query" << std::endl;
      if ((t1 - t0) > testConfig.maxTimeThreshold)
        break;
      numPerRun*=2;
    };
  }
}

using namespace testing;

int main(int ac, char **av)
{
  BuildConfig buildConfig;
  std::string bvhType = "binary";
  TestConfig testConfig;
  int numDims = 3;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-d" || arg == "--data") {
      testConfig.dataDist = av[++i];
      testConfig.dataCount = std::stoi(av[++i]);
    } else if (arg == "-q" || arg == "--query") {
      testConfig.queryDist = av[++i];
      testConfig.queryCount = std::stoi(av[++i]);
    } else if (arg == "-nd" || arg == "--num-dims") {
      if (av[i+1] == "n")
        numDims = CUBQL_TEST_N;
      else
        numDims = std::stoi(av[++i]);
    } else if (arg == "-bt" || arg == "--bvh-type")
      bvhType = av[++i];
    else if (arg == "-sah" || arg == "--sah")
      buildConfig.enableSAH();
    else if (arg == "-mr" || arg == "-mqr" || arg == "--max-query-radius")
      testConfig.maxQueryRadius = std::stof(av[++i]);
    else if (arg == "--check-reference") {
      testConfig.referenceFileName = av[++i];
      testConfig.make_reference = false;
    } else if (arg == "--make-reference") {
      testConfig.referenceFileName = av[++i];
      testConfig.make_reference = true;
      testConfig.maxTimeThreshold = 0.f;
    } else if (arg == "-mlt" || arg == "-lt")
      buildConfig.makeLeafThreshold = std::stoi(av[++i]);
    else
      usage("unknown cmd-line argument '"+arg+"'");
  }
  // std::vector<float3> dataPoints  = loadData<float3>(generators[0]);
  // std::vector<float3> queryPoints = loadData<float3>(generators[1]);

  // reference = loadData<float>(referenceFileName);
  if (numDims == 2)
    testFCP<2>(testConfig,buildConfig);
  else if (numDims == 3)
    testFCP<3>(testConfig,buildConfig);
  else if (numDims == 4)
    testFCP<4>(testConfig,buildConfig);
#if CUBQL_TEST_N
  else if (numDims == CUBQL_TEST_N)
    testFCP<CUBQL_TEST_N>(testConfig,buildConfig);
#endif
  else
    throw std::runtime_error("unsupported number of dimensions "+std::to_string(numDims));
  // if (!referenceFileName.empty() && !make_reference)
  //   reference = loadData<float>(referenceFileName);

  // if (bvhType == "binary")
  //   testing::testFCP<BinaryBVH<float,3>>(dataPoints,queryPoints,buildConfig,testConfig);
  // // else if (bvhType == "bvh2")
  // //   testing::testFCP<WideBVH<float,3,2>>(dataPoints,queryPoints,buildConfig,testConfig);
  // else if (bvhType == "bvh4")
  //   testing::testFCP<WideBVH<float,3,4>>(dataPoints,queryPoints,buildConfig,testConfig);
  // else if (bvhType == "bvh8")
  //   testing::testFCP<WideBVH<float,3,8>>(dataPoints,queryPoints,buildConfig,testConfig);
  // else
  //   throw std::runtime_error("unknown or not-yet-hooked-up bvh type '"+bvhType+"'");

  // if (make_reference) {
  //   std::cout << "saving reference data to " << referenceFileName << std::endl;
  //   saveData(reference,referenceFileName);
  return 0;
}
