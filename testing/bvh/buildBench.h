// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

/*! \file buildBench.h Implements simple benchmark app for bvh
    construction */
#pragma once

// #define CUBQL_GPU_BUILDER_IMPLEMENTATION

#include "cuBQL/bvh.h"
#include "samples/common/CmdLine.h"
#include "samples/common/IO.h"
#include "testing/common/testRig.h"
#include "samples/common/Generator.h"

namespace testing {

  typedef enum {
    BUILDTYPE_DEFAULT=0,
    BUILDTYPE_RADIX,
    BUILDTYPE_REBIN,
    BUILDTYPE_SAH }
    BuildType;
  using namespace cuBQL;
  using namespace cuBQL::samples;
      
  using vecND   = cuBQL::vec_t<double,CUBQL_TEST_D>;
  using box_t   = cuBQL::box_t<CUBQL_TEST_T,CUBQL_TEST_D>;
  using point_t = cuBQL::vec_t<CUBQL_TEST_T,CUBQL_TEST_D>;
  using bvh_t   = cuBQL::bvh_t<CUBQL_TEST_T,CUBQL_TEST_D>;

  void computeBoxes(box_t *d_boxes,
                    const point_t *d_data,
                    int numData);
  bvh_t computeBVH(const box_t *d_boxes,
                   int numBoxes,
                   BuildType buildType);
  void free(bvh_t bvh);

  void usage(const std::string &error = "")
  {
    if (!error.empty())
      std::cout << "Error: " << error << "\n\n";
    std::cout << "Usage: ./cuBQL...buildBench... -n <numPoints> [--clustered|-c] [--uniform|-u]" << std::endl;
    exit(error.empty()?0:1);
  }
      
  void main(int ac, char **av,
            cuBQL::testRig::DeviceAbstraction &device)
  {
    bool        useClustered = false;
    int         numPoints = 10000000;
    CmdLine     cmdLine(ac,av);
    BuildType   buildType = BUILDTYPE_DEFAULT;
    std::string inFileName;
    while (!cmdLine.consumed()) {
      const std::string arg = cmdLine.getString();
      if (arg == "-u" || arg == "--uniform")
        useClustered = false;
      else if (arg == "-c" || arg == "--clustered")
        useClustered = true;
      else if (arg == "-n") 
        numPoints = cmdLine.getInt();
      else if (arg == "--radix" || arg == "--morton")
        buildType = BUILDTYPE_RADIX;
      else if (arg == "--rebin")
        buildType = BUILDTYPE_REBIN;
      else if (arg == "--sah")
        buildType = BUILDTYPE_SAH;
      else if (arg[0] != '-')
        inFileName = arg;
      else 
        usage("un-recognized cmd-line argument '"+arg+"'");
    }

    std::vector<vecND> generatedPoints;
    if (inFileName.empty()) {
      std::cout << "generating " << numPoints
                << (useClustered?" clustered":" uniformly distributed")
                << " data points" << std::endl;
      generatedPoints
        = useClustered
        ? cuBQL::samples::ClusteredPointGenerator<CUBQL_TEST_D>().generate(numPoints,12345)
        : cuBQL::samples::UniformPointGenerator<CUBQL_TEST_D>().generate(numPoints,12345);
      
      std::cout << "converting to " << point_t::typeName() << std::endl;
    } else {
      generatedPoints = loadBinary<vecND>(inFileName);
    }
    std::vector<point_t> dataPoints
      = cuBQL::samples::convert<CUBQL_TEST_T>(generatedPoints);
    
    const point_t *d_dataPoints
      = device.upload(dataPoints);
    int numData = int(dataPoints.size());
        
    std::cout << "computing boxes for bvh build" << std::flush << std::endl;
    int numBoxes = numData;
    box_t *d_boxes = device.alloc<box_t>(numBoxes);
    computeBoxes(d_boxes,d_dataPoints,numData);
        
    std::cout << "computing bvh - first time for warmup" << std::flush << std::endl;
    bvh_t bvh = computeBVH(d_boxes,numBoxes,buildType);
    free(bvh);

    int minMeasureCount = 10;
    double minMeasureTime /* in sec */ = 10.f;
    std::cout << "now doing up to " << minMeasureCount
              << " re-builds, or re-builds for " << prettyDouble(minMeasureTime)
              << "s, whichever comes first" << std::endl;
    double t0 = getCurrentTime(), t1;
    int numBuildsDone = 0;
    while (true) {
      bvh = computeBVH(d_boxes,numBoxes,buildType);
      free(bvh);
      numBuildsDone++;
      t1 = getCurrentTime();
      if (numBuildsDone >= minMeasureCount && (t1-t0) >= minMeasureTime)
        break;
    }
    std::cout << "done " << numBuildsDone << " in " << prettyDouble(t1-t0) << "s" << std::endl;
    std::cout << "--- total builds per second" << std::endl;
    std::cout << "    BUILDS_PER_SECOND "
              << prettyDouble(numBuildsDone/(t1-t0)) << std::endl;
    std::cout << "--- avg time per build" << std::endl;
    std::cout << "    AVG_TIME_PER_BUILD "
              << prettyDouble((t1-t0)/numBuildsDone) << std::endl;
    std::cout << "--- build speed in prims per second (for given prim count)" << std::endl;
    std::cout << "    PRIMS_PER_SECOND "
              << prettyDouble(numPoints*(size_t)numBuildsDone/(t1-t0)) << std::endl;
    
    device.free(d_dataPoints);
  }

} // ::testing

