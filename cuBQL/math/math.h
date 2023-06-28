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

#include "cuBQL/math/common.h"

namespace cuBQL {

  using ::min;
  using ::max;

  using ::make_float3;
  
  template<int N> struct log_of;
  template<> struct log_of< 2> { enum { value = 1 }; };
  template<> struct log_of< 4> { enum { value = 2 }; };
  template<> struct log_of< 8> { enum { value = 3 }; };
  template<> struct log_of<16> { enum { value = 4 }; };

  /*! square of a value */
  inline __both__ float sqr(float f) { return f*f; }
}

