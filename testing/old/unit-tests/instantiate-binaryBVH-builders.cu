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

template<typename T>
void foo()
{
  using box_t = cuBQL::box_t<T,UNIT_TEST_N_FROM_CMAKE>;
  using bvh_t = cuBQL::bvh_t<T,UNIT_TEST_N_FROM_CMAKE>;
  using bvh4_t = cuBQL::WideBVH<T,UNIT_TEST_N_FROM_CMAKE,4>;
  using bvh8_t = cuBQL::WideBVH<T,UNIT_TEST_N_FROM_CMAKE,8>;

  /* this obviously will not RUN, but it should at least trigger the
     template instantiation */
  box_t *d_boxes = 0;
  int numBoxes = 0;

  cuBQL::BuildConfig       buildConfig;
  
  bvh_t bvh;
  gpuBuilder(bvh,d_boxes,numBoxes,buildConfig);
  
  bvh4_t bvh4;
  gpuBuilder(bvh4,d_boxes,numBoxes,buildConfig);
  
  bvh8_t bvh8;
  gpuBuilder(bvh8,d_boxes,numBoxes,buildConfig);
}

int main(int, char **)
{
  foo<float>();
  return 0;
}
