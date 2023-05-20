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

#pragma once

#if CUBQL_GPU_BUILDER_IMPLEMENTATION
#include "cuBQL/impl/builder_common.h"
#include "cuBQL/impl/sm_builder.h"
#include "cuBQL/impl/sah_builder.h"

namespace cuBQL {

  void gpuBuilder(BinaryBVH   &bvh,
                  const box3f *boxes,
                  uint32_t     numBoxes,
                  BuildConfig  buildConfig,
                  cudaStream_t s)
  {
    if (buildConfig.makeLeafThreshold == 0)
      // unless explicitly specified, use default for spatial median
      // builder:
      buildConfig.makeLeafThreshold = 8;
    if (buildConfig.buildMethod == BuildConfig::SAH)
      sahBuilder_impl::sahBuilder(bvh,boxes,numBoxes,buildConfig,s);
    else
      gpuBuilder_impl::build(bvh,boxes,numBoxes,buildConfig,s);
    gpuBuilder_impl::refit(bvh,boxes,s);
    CUBQL_CUDA_CALL(StreamSynchronize(s));
  }

  float computeSAH(const BinaryBVH &bvh)
  {
    return gpuBuilder_impl::computeSAH(bvh);
  }
  
  void free(BinaryBVH   &bvh,
            cudaStream_t s)
  {
    CUBQL_CUDA_CALL(StreamSynchronize(s));
    CUBQL_CUDA_CALL(FreeAsync(bvh.primIDs,s));
    CUBQL_CUDA_CALL(FreeAsync(bvh.nodes,s));
    CUBQL_CUDA_CALL(StreamSynchronize(s));
    bvh.primIDs = 0;
  }
}
#endif

