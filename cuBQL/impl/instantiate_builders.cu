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

/*! instantiates the GPU builder(s) */
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"

namespace cuBQL {
  // template void gpuBuilder(WideBVH<2>   &bvh,
  //                          const box3f *boxes,
  //                          uint32_t     numBoxes,
  //                          BuildConfig  buildConfig,
  //                          cudaStream_t s);
  template void gpuBuilder(WideBVH<box3f,4>   &bvh,
                           const box3f *boxes,
                           uint32_t     numBoxes,
                           BuildConfig  buildConfig,
                           cudaStream_t s);
  template void gpuBuilder(WideBVH<box3f,8>   &bvh,
                           const box3f *boxes,
                           uint32_t     numBoxes,
                           BuildConfig  buildConfig,
                           cudaStream_t s);
  // template void gpuBuilder(WideBVH<16>   &bvh,
  //                          const box3f *boxes,
  //                          uint32_t     numBoxes,
  //                          BuildConfig  buildConfig,
  //                          cudaStream_t s);

  // template void free(WideBVH<2>  &bvh, cudaStream_t s);
  template void free(WideBVH<box3f,4>  &bvh, cudaStream_t s);
  template void free(WideBVH<box3f,8>  &bvh, cudaStream_t s);
  // template void free(WideBVH<16> &bvh, cudaStream_t s);
  
  // template float computeSAH(const WideBVH<2>  &bvh);
  template float computeSAH(const WideBVH<box3f,4>  &bvh);
  template float computeSAH(const WideBVH<box3f,8>  &bvh);
  // template float computeSAH(const WideBVH<16> &bvh);
  
}
