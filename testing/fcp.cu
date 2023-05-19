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

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/fcp.h"

#include "cuBQL/CUDAArray.h"
#include "testing/helper.h"

namespace testing {

  using box_t = cuBQL::box3f;

  void usage(const std::string &error = "")
  {
    if (!error.empty()) {
      std::cerr << error << "\n\n";
    }
    std::cout << "./cuBQL_fcp dataPoints.dat queryPoints.dat\n\n";
    exit(error.empty()?0:1);
  }

  __global__ void makeBoxes(box_t *boxes, float3 *points, int numPoints)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;
    float3 point = points[tid];
    boxes[tid].lower = point;
    boxes[tid].upper = point;
  }

  __global__ void resetResults(int *results, int numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;
    results[tid] = -1;
  }

  __global__ void runFCP(int *results,
                         BinaryBVH bvh,
                         const float3 *dataPoints,
                         const float3 *queries,
                         int numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;

    results[tid] = cuBQL::fcp(bvh,dataPoints,queries[tid]);
  }
  
  void testFCP(const std::vector<float3> &h_dataPoints,
               const std::vector<float3> &h_queryPoints,
               int maxLeafSize,
               float maxTimeThreshold = 10.f
               )
  {
    cuBQL::CUDAArray<float3> dataPoints;
    dataPoints.upload(h_dataPoints);
    cuBQL::CUDAArray<box_t> boxes(dataPoints.size());
    {
      int bs = 256;
      int nb = divRoundUp((int)dataPoints.size(),bs);
      makeBoxes<<<nb,bs>>>(boxes.data(),dataPoints.data(),(int)dataPoints.size());
    };
    
    cuBQL::BinaryBVH bvh;
    cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),maxLeafSize);

    cuBQL::CUDAArray<float3> queryPoints;
    queryPoints.upload(h_queryPoints);
    
    int numQueries = queryPoints.size();
    cuBQL::CUDAArray<int> closest(numQueries);

    int numPerRun = 1;
    while (true) {
      CUBQL_CUDA_SYNC_CHECK();
      double t0 = getCurrentTime();
      for (int i=0;i<numPerRun;i++) {
        resetResults<<<divRoundUp(numQueries,128),128>>>(closest.data(),numQueries);
        runFCP<<<divRoundUp(numQueries,128),128>>>
          (closest.data(),
           bvh,
           dataPoints.data(),
           queryPoints.data(),
           numQueries);
        CUBQL_CUDA_SYNC_CHECK();
      }
      double t1 = getCurrentTime();
      std::cout << "done " << numPerRun
                << " queries in " << prettyDouble(t1-t0) << "s, that's "
                << prettyDouble((t1-t0)/numPerRun) << "s query" << std::endl;
      if ((t1 - t0) > maxTimeThreshold)
        break;
      numPerRun*=2;
    };
  }
}

using namespace testing;

int main(int ac, char **av)
{
  int maxLeafSize = 8;
  std::vector<std::string> fileNames;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (av[i][0] != '-')
      fileNames.push_back(arg);
      else if (arg == "-mls" || arg == "-ls")
        maxLeafSize = std::stoi(av[++i]);
      else
        usage("unknown cmd-line argument '"+arg+"'");
  }
  if (fileNames.size() != 2)
    usage("unexpected number of data file names");
  std::vector<float3> dataPoints  = loadData<float3>(fileNames[0]);
  std::vector<float3> queryPoints = loadData<float3>(fileNames[1]);
  
  testing::testFCP(dataPoints,queryPoints,maxLeafSize);
  return 0;
}
