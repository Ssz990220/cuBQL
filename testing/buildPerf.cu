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

#include "testing/helper/CUDAArray.h"
#include "testing/helper.h"

namespace cuBQL {
  namespace test_rig {

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
                   cuBQL::BuildConfig buildConfig,
                   float numSecsAvg)
    {
      int numPrims = h_boxes.size();
      std::cout << "measuring build performance for "
                << prettyNumber(numPrims) << " prims" << std::endl;
      CUDAArray<box_t> boxes;
      boxes.upload(h_boxes);

      std::cout << "... initial warm-up build" << std::endl;
      bvh_t bvh;
      cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),buildConfig);
      if (bvh_t::numDims == 3) 
        std::cout << "done build, sah cost is " << cuBQL::computeSAH(bvh) << std::endl;
      else
        std::cout << "done build..." << std::endl;
      
      double t0 = getCurrentTime();
      int thisRunSize = 1;
      while (true) {
        std::cout << "- going to run build " << thisRunSize << " time(s) in succession:" << std::endl;
        double t0 = getCurrentTime();
        for (int i=0;i<thisRunSize;i++) {
          cuBQL::free(bvh);
          cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),buildConfig);
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
  } // ::cuBQL::test_rig
} // ::cuBQL

using namespace cuBQL::test_rig;

int main(int ac, char **av)
{
  cuBQL::BuildConfig buildConfig;
  float numSecsAvg = 5.f;
  std::string bvhType = "binary";
  std::vector<std::string> fileNames;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (av[i][0] != '-')
      fileNames.push_back(arg);
    else if (arg == "-mlt" || arg == "-lt")
      buildConfig.makeLeafThreshold = std::stoi(av[++i]);
    else if (arg == "-ns")
      numSecsAvg = std::stof(av[++i]);
    else if (arg == "--bvh-type")
      bvhType = av[++i];
    else if (arg == "-sah")
      buildConfig.buildMethod = buildConfig.SAH;
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
  std::vector<vec3f> dataPoints  = loadData<vec3f>(fileNames[0]);
  for (auto point : dataPoints) 
    boxes.push_back(box3f{point,point});
#endif

  if (bvhType == "binary")
    cuBQL::test_rig::buildPerf<cuBQL::BinaryBVH<float,3>>
      (boxes,buildConfig,numSecsAvg);
  // else if (bvhType == "bvh2")
  // cuBQL::test_rig::buildPerf<cuBQL::WideBVH<float,3,2>>
  //   (boxes,buildConfig,numSecsAvg);
  else if (bvhType == "bvh4")
    cuBQL::test_rig::buildPerf<cuBQL::WideBVH<float,3,4>>
      (boxes,buildConfig,numSecsAvg);
  else if (bvhType == "bvh8")
    cuBQL::test_rig::buildPerf<cuBQL::WideBVH<float,3,8>>
      (boxes,buildConfig,numSecsAvg);
  // else if (bvhType == "bvh16")
  //   cuBQL::test_rig::buildPerf<cuBQL::WideBVH<float,3,16>>
  //     (boxes,buildConfig,numSecsAvg);
  else
    throw std::runtime_error("unsupported bvh type '"+bvhType+"'");
  return 0;
}
