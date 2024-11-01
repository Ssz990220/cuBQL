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
#include "cuBQL/builder/cuda.h"

namespace testing {

  __global__
  void computeBox(box_t *d_boxes, const point_t *d_data, int numData)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numData) return;

    d_boxes[tid] = box_t().including(d_data[tid]);
  }
      
  void computeBoxes(box_t *d_boxes,
                    const point_t *d_data,
                    int numData)
  {
    computeBox<<<divRoundUp(numData,128),128>>>(d_boxes,d_data,numData);
    CUBQL_CUDA_SYNC_CHECK();
  }
      
  bvh_t computeBVH(const box_t *d_boxes,
                   int numBoxes)
  {
    bvh_t bvh;
    cuBQL::gpuBuilder(bvh,d_boxes,numBoxes,BuildConfig());
    return bvh;
  }

  __global__
  void runQueries(bvh_t bvh,
                  const point_t  *d_data,
                  float      *d_results,
                  const point_t *d_queries,
                  int            numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;
        
    d_results[tid] = runQuery(bvh,d_data,d_queries[tid]);
  }

  void launchQueries(bvh_t bvh,
                     const point_t  *d_data,
                     float      *d_results,
                     const point_t *d_queries,
                     int            numQueries)
  {
    runQueries<<<divRoundUp(numQueries,128),128>>>
      (bvh,d_data,d_results,d_queries,numQueries);
  }
      
  void computeReferenceResults(const point_t  *d_data,
                               int            numData,
                               float      *d_results,
                               const point_t *d_queries,
                               int            numQueries)
  {
    throw std::runtime_error("computing reference results only implemented on host");
  }
  
  void free(bvh_t bvh)
  { cuBQL::cuda::free(bvh); }
  
} // ::testing

int main(int ac, char **av)
{
  cuBQL::testRig::CUDADevice device;
  testing::main(ac,av,device);
  return 0;
}
