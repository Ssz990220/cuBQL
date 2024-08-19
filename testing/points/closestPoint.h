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

#pragma once

#include "cuBQL/bvh.h"
#include "cuBQL/queries/points/findClosest.h"
#include "samples/common/CmdLine.h"
#include "samples/common/IO.h"
#include "testing/common/testRig.h"

namespace cuBQL {
  namespace points {
    namespace closestPoint {
      using namespace cuBQL::samples;
      
      using box_t = cuBQL::box_t<CUBQL_TEST_T,CUBQL_TEST_D>;
      using data_t = cuBQL::vec_t<CUBQL_TEST_T,CUBQL_TEST_D>;
      using query_t = cuBQL::vec_t<CUBQL_TEST_T,CUBQL_TEST_D>;
      using result_t = float;
      using bvh_t = cuBQL::bvh_t<CUBQL_TEST_T,CUBQL_TEST_D>;

      inline __cubql_both
      float runQuery(bvh_t bvh,
                     const data_t *data,
                     query_t query)
      {
        int closestID
          = cuBQL::points::findClosest(bvh,data,query);
        if (closestID < 0)
          return INFINITY;
        data_t closest = data[closestID];
        return sqrDistance(closest,query);
      }
        
                                   
      void computeBoxes(box_t *d_boxes,
                        const data_t *d_data,
                        int numData);
      bvh_t computeBVH(const box_t *d_boxes,
                       int numBoxes);
      void launchQueries(bvh_t bvh,
                         const data_t  *d_data,
                         result_t      *d_results,
                         const query_t *d_queries,
                         int            numQueries);
      void free(bvh_t bvh);
      
      void main(int ac, char **av,
                cuBQL::testRig::DeviceAbstraction &device)
      {
        std::string dataFileName = "<not_specified>";
        std::string queryFileName = "<not_specified>";
        std::string goldFileName = "<not_specified>";
        bool rebuildGold = false;

        CmdLine cmdLine(ac,av);
        while (!cmdLine.consumed()) {
          const std::string arg = cmdLine.getString();
          if (arg == "-g")
            goldFileName = cmdLine.getString();
          else if (arg == "-d")
            dataFileName = cmdLine.getString();
          else if (arg == "-q")
            queryFileName = cmdLine.getString();
          else if (arg == "--rebuild-gold")
            rebuildGold = true;
          else
            throw std::runtime_error("Usage: "+std::string(av[0])+" -d dataPoints.bin -q queryPoints.bin -g goldResults.bin [--rebuild-gold]");
        }
        std::vector<data_t>  dataPoints  = loadBinary<data_t>(dataFileName);
        std::vector<query_t> queryPoints = loadBinary<query_t>(queryFileName);
        
        const data_t *d_dataPoints
          = device.upload(dataPoints);
        int numData = int(dataPoints.size());
        
        const query_t *d_queries
          = device.upload(queryPoints);
        int numQueries = int(queryPoints.size());
        
        result_t *d_results
          = device.alloc<result_t>(queryPoints.size());

        std::cout << "computing boxes for bvh build" << std::flush << std::endl;
        int numBoxes = numData;
        box_t *d_boxes = device.alloc<box_t>(numBoxes);
        computeBoxes(d_boxes,d_dataPoints,numData);
        
        std::cout << "computing bvh" << std::flush << std::endl;
        bvh_t bvh = computeBVH(d_boxes,numBoxes);
        
        std::cout << "launching queries" << std::flush << std::endl;
        launchQueries(bvh,d_dataPoints,
                      d_results,d_queries,numQueries);
        std::cout << "queries done, downloading results..." << std::flush << std::endl;
        std::vector<result_t> results = device.download(d_results,queryPoints.size());
        if (rebuildGold) {
          std::cout << "#cuBQL: asked to _rebuild_ gold results, "
                    << "so just saving instead of comparing" << std::endl;
          saveBinary(goldFileName,results);
        } else {
          std::cout << "loading reference 'gold' results..." << std::flush << std::endl;
          std::vector<result_t> gold = loadBinary<result_t>(goldFileName);
          for (int i=0;i<(int)results.size();i++) {
            if (results[i] != gold[i])
              std::cout << "!! different result for index " << i << ": ours says " << results[i] << ", gold says " << gold[i] << std::endl;
          }
        }
        device.free(d_results);
        device.free(d_dataPoints);
        device.free(d_queries);
      }
    }
  }
}

