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

  // typedef cuBQL::box3f box_t;
  using box_t = cuBQL::box3f;

  void usage(const std::string &error = "")
  {
    if (!error.empty()) {
      std::cerr << error << "\n\n";
    }
    std::cout << "./buildPerf_points dataPoints.dat\n\n";
    exit(error.empty()?0:1);
  }

  template<typename bvh_t>
  void buildPerf(const std::vector<box_t> &h_boxes,
                 int maxLeafSize,
                 bool sah,
                 float numSecsAvg)
  {
    int numPrims = h_boxes.size();
    std::cout << "measuring build performance for "
              << prettyNumber(numPrims) << " prims" << std::endl;
    cuBQL::CUDAArray<box_t> boxes;
    boxes.upload(h_boxes);

    std::cout << "... initial warm-up build" << std::endl;
    bvh_t bvh;
    if (sah)
      cuBQL::gpuSAHBuilder(bvh,boxes.data(),boxes.size(),maxLeafSize);
    else
      cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),maxLeafSize);
    std::cout << "done build, sah cost is " << cuBQL::computeSAH(bvh) << std::endl;
    double t0 = getCurrentTime();
    int thisRunSize = 1;
    while (true) {
      std::cout << "- going to run build " << thisRunSize << " time(s) in succession:" << std::endl;
      double t0 = getCurrentTime();
      for (int i=0;i<thisRunSize;i++) {
        cuBQL::free(bvh);
        if (sah)
          cuBQL::gpuSAHBuilder(bvh,boxes.data(),boxes.size(),maxLeafSize);
        else
          cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),maxLeafSize);
      }
      double t1 = getCurrentTime();
      
      double buildsPerSecond = thisRunSize / (t1-t0);
      double primsPerSecond = (thisRunSize*(double)numPrims) / (t1-t0);
      std::cout << " ...done " << thisRunSize << " builds in "
                << prettyDouble(t1-t0) << "s, that's "
                << prettyDouble(buildsPerSecond) << " builds/s (or "
                << prettyDouble(1.f/buildsPerSecond) << "s/build); or "
                << prettyDouble(primsPerSecond) << "prims/s" << std::endl;

      if ((t1 - t0) > numSecsAvg) {
        std::cout << "MSPB " << int(1000*(t1-t0)/thisRunSize+.5f) << std::endl;
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
  bool sah = false;
  std::string bvhType = "binary";
  std::vector<std::string> fileNames;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (av[i][0] != '-')
      fileNames.push_back(arg);
    else if (arg == "-mls" || arg == "-ls")
      maxLeafSize = std::stoi(av[++i]);
    else if (arg == "-ns")
      numSecsAvg = std::stof(av[++i]);
    else if (arg == "--bvh-type")
      bvhType = av[++i];
    else if (arg == "-sah")
      sah = true;
    else
      usage("unknown cmd-line argument '"+arg+"'");
    }
    if (fileNames.size() != 1)
      usage("unexpected number of data file names");
    std::vector<box3f> boxes;
#if EXPECT_TRIS
    boxes = loadData<box3f>(fileNames[0]);
#endif
#if EXPECT_POINTS
    std::vector<float3> dataPoints  = loadData<float3>(fileNames[0]);
    for (auto point : dataPoints) 
      boxes.push_back({point,point});
#endif

    if (bvhType == "binary")
      testing::buildPerf<cuBQL::BinaryBVH>
        (boxes,maxLeafSize,sah,numSecsAvg);
    else if (bvhType == "bvh2")
      testing::buildPerf<cuBQL::WideBVH<2>>
        (boxes,maxLeafSize,sah,numSecsAvg);
    else if (bvhType == "bvh4")
      testing::buildPerf<cuBQL::WideBVH<4>>
        (boxes,maxLeafSize,sah,numSecsAvg);
    else if (bvhType == "bvh8")
      testing::buildPerf<cuBQL::WideBVH<8>>
        (boxes,maxLeafSize,sah,numSecsAvg);
    else if (bvhType == "bvh16")
      testing::buildPerf<cuBQL::WideBVH<16>>
        (boxes,maxLeafSize,sah,numSecsAvg);
    else
      throw std::runtime_error("unsupported bvh type '"+bvhType+"'");
    return 0;
  }
