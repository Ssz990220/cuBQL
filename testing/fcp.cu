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
#include "cuBQL/queries/fcp.h"

#include "testing/helper/CUDAArray.h"
#include "testing/helper.h"

namespace testing {

  using box_t = cuBQL::box3f;

  std::vector<float> reference;
  
  struct TestConfig {
    float maxTimeThreshold = 10.f;
    float maxQueryRadius = INFINITY;
  };
  
  void usage(const std::string &error = "")
  {
    if (!error.empty()) {
      std::cerr << error << "\n\n";
    }
    std::cout << "./cuBQL_fcp dataPoints.dat queryPoints.dat\n\n";
    exit(error.empty()?0:1);
  }

  __global__
  void makeBoxes(box_t *boxes, float3 *points, int numPoints)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;
    float3 point = points[tid];
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
  
  template<typename bvh_t>
  __global__
  void runFCP(float        *results,
              bvh_t         bvh,
              const float3 *dataPoints,
              const float3 *queries,
              float         maxRadius,
              int           numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;

    const float3 query = queries[tid];
    float sqrMaxQueryDist = maxRadius*maxRadius;
    int res = cuBQL::fcp(bvh,dataPoints,query,
                         /* fcp kernel expects SQUARE of radius */
                         sqrMaxQueryDist);
    if (res == -1)
      results[tid] = INFINITY;
    else
      // results[tid] = sqrDistance(dataPoints[res],query);
      results[tid] = sqrMaxQueryDist;
  }

  template<typename bvh_t>
  void testFCP(const std::vector<float3> &h_dataPoints,
               const std::vector<float3> &h_queryPoints,
               BuildConfig buildConfig,
               TestConfig testConfig
               )
  {
    CUDAArray<float3> dataPoints;
    dataPoints.upload(h_dataPoints);

    CUDAArray<box_t> boxes(dataPoints.size());
    {
      int bs = 256;
      int nb = divRoundUp((int)dataPoints.size(),bs);
      makeBoxes<<<nb,bs>>>(boxes.data(),dataPoints.data(),(int)dataPoints.size());
    };

    // cuBQL::BinaryBVH
    bvh_t bvh;
    cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),buildConfig);
    std::cout << "bvh is built, SAH cost is " << cuBQL::computeSAH(bvh) << std::endl;

    CUDAArray<float3> queryPoints;
    queryPoints.upload(h_queryPoints);
    
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
  std::vector<std::string> fileNames;
  std::string bvhType = "binary";
  bool make_reference = false;
  std::string referenceFileName;
  TestConfig testConfig;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (av[i][0] != '-')
      fileNames.push_back(arg);
    else if (arg == "-bt" || arg == "--bvh-type")
      bvhType = av[++i];
    else if (arg == "-sah" || arg == "--sah")
      buildConfig.enableSAH();
    else if (arg == "-mr" || arg == "-mqr" || arg == "--max-query-radius")
      testConfig.maxQueryRadius = std::stof(av[++i]);
    else if (arg == "--check-reference") {
      referenceFileName = av[++i];
      make_reference = false;
    } else if (arg == "--make-reference") {
      referenceFileName = av[++i];
      make_reference = true;
      testConfig.maxTimeThreshold = 0.f;
    } else if (arg == "-mlt" || arg == "-lt")
      buildConfig.makeLeafThreshold = std::stoi(av[++i]);
    else
      usage("unknown cmd-line argument '"+arg+"'");
  }
  if (fileNames.size() < 2)
    usage("unexpected number of data file names");
  std::vector<float3> dataPoints  = loadData<float3>(fileNames[0]);
  std::vector<float3> queryPoints = loadData<float3>(fileNames[1]);

  if (!referenceFileName.empty() && !make_reference)
    reference = loadData<float>(referenceFileName);

  if (bvhType == "binary")
    testing::testFCP<BinaryBVH>(dataPoints,queryPoints,buildConfig,testConfig);
  else if (bvhType == "bvh2")
    testing::testFCP<WideBVH<2>>(dataPoints,queryPoints,buildConfig,testConfig);
  else if (bvhType == "bvh4")
    testing::testFCP<WideBVH<4>>(dataPoints,queryPoints,buildConfig,testConfig);
  else if (bvhType == "bvh8")
    testing::testFCP<WideBVH<8>>(dataPoints,queryPoints,buildConfig,testConfig);
  else
    throw std::runtime_error("unknown or not-yet-hooked-up bvh type '"+bvhType+"'");

  if (make_reference) {
    std::cout << "saving reference data to " << referenceFileName << std::endl;
    saveData(reference,referenceFileName);
  }
  return 0;
}
