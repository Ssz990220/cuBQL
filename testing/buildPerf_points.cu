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

#include "cuBQL/CUDAArray.h"
#include "testing/helper.h"

namespace testing {

  typedef cuBQL::box3fa box_t;

  void usage(const std::string &error = "")
  {
    if (!error.empty()) {
      std::cerr << error << "\n\n";
    }
    std::cout << "./buildPerf_points dataPoints.dat\n\n";
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

  void buildPerf(const std::vector<float3> &h_dataPoints,
                 int maxLeafSize,
                 float numSecsAvg)
  {
    int numPrims = h_dataPoints.size();
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

    double t0 = getCurrentTime();
    int thisRunSize = 1;
    while (true) {
      double t0 = getCurrentTime();
      for (int i=0;i<thisRunSize;i++) {
        cuBQL::free(bvh);
        cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),maxLeafSize);
      }
      double t1 = getCurrentTime();
      if ((t1 - t0) > numSecsAvg) {
        double buildsPerSecond = thisRunSize / (t1-t0);
        double primsPerSecond = (thisRunSize*(double)numPrims) / (t1-t0);
        std::cout << "done. re-built BVH " << thisRunSize << " times in " << (t1-t0) << " secs, that's " << prettyDouble(buildsPerSecond) << " builds per second (or " << prettyDouble(1.f/buildsPerSecond) << "s/build); or " << prettyDouble(primsPerSecond) << "prims/sec" << std::endl;
        std::cout << "BPS " << buildsPerSecond << std::endl;
        std::cout << "PPS " << primsPerSecond << std::endl;
        break;
      }
      thisRunSize *= 2;
    }
    cuBQL::free(bvh);
  }
}

using namespace testing;

int main(int ac, char **av)
{
  int maxLeafSize = 8;
  float numSecsAvg = 5.f;
  std::vector<std::string> fileNames;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (av[i][0] != '-')
      fileNames.push_back(arg);
    else if (arg == "-mls" || arg == "-ls")
      maxLeafSize = std::stoi(av[++i]);
    else if (arg == "-ns")
      numSecsAvg = std::stof(av[++i]);
    else
      usage("unknown cmd-line argument '"+arg+"'");
    }
    if (fileNames.size() != 1)
      usage("unexpected number of data file names");
    std::vector<float3> dataPoints  = loadData<float3>(fileNames[0]);

    testing::buildPerf(dataPoints,maxLeafSize,numSecsAvg);
    return 0;
  }
