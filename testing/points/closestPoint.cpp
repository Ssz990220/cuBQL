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

#include "closestPoint.h"
#include "cuBQL/builder/host.h"

namespace testing {

  void computeBoxes(box_t *d_boxes,
                    const data_t *d_data,
                    int numData)
  {
    for (int tid=0; tid<numData; tid++)
      d_boxes[tid] = box_t().including(d_data[tid]);
  }
      
  bvh_t computeBVH(const box_t *d_boxes, int numBoxes)
  {
    bvh_t bvh;
    cuBQL::host::spatialMedian(bvh,d_boxes,numBoxes,BuildConfig());
    return bvh;
  }
      
  void computeReferenceResults(const data_t  *d_data,
                               int            numData,
                               result_t      *d_results,
                               const query_t *d_queries,
                               int            numQueries)
  {
    for (int qi=0;qi<numQueries;qi++) {
      query_t query = d_queries[qi];
      cuBQL::vec_t<double,CUBQL_TEST_D>
        doubleQuery = cuBQL::vec_t<double,CUBQL_TEST_D>(query);
      if (qi == 0)
        PRINT(doubleQuery);
      float closest = INFINITY;
      for (int di=0;di<numData;di++) {
        cuBQL::vec_t<double,CUBQL_TEST_D>
          doubleData = cuBQL::vec_t<double,CUBQL_TEST_D>(d_data[di]);
        cuBQL::vec_t<double,CUBQL_TEST_D>
          diff = doubleData - doubleQuery;

        if (qi == 0) {
          PRINT(di);
          PRINT(doubleData);
        }
        
        float d = dot(diff,diff);
        closest = std::min(closest,d);
      }
      d_results[qi] = closest;
    }
  }
  
  void launchQueries(bvh_t bvh,
                     const data_t  *d_data,
                     result_t      *d_results,
                     const query_t *d_queries,
                     int            numQueries)
  {
    for (int tid=0;tid<numQueries;tid++)
      d_results[tid] = runQuery(bvh,d_data,d_queries[tid]);
  }

  void free(bvh_t bvh)
  { cuBQL::host::freeBVH(bvh); }

} // ::testing

int main(int ac, char **av)
{
  cuBQL::testRig::HostDevice device;
  testing::main(ac,av,device);
  return 0;
}
