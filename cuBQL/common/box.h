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

#include "cuBQL/common/vec.h"

namespace cuBQL {

  template<typename scalar_t>
  inline __both__ scalar_t empty_box_lower_value();
  template<typename scalar_t>
  inline __both__ scalar_t empty_box_upper_value();

  template<> inline __both__ float empty_box_lower_value<float>() { return +INFINITY; }
  template<> inline __both__ float empty_box_upper_value<float>() { return -INFINITY; }
  
  
  template<typename vec_t>
  struct box_t {
    enum { numDims = vec_traits<vec_t>::numDims };
    using scalar_t = typename vec_traits<vec_t>::scalar_t;
    using cuda_vec_t = typename cuda_eq_t<scalar_t,numDims>::type;
    
    inline __both__ box_t &grow(vec_t v)
    { lower = min(lower,v); upper = max(upper,v); return *this; }
    inline __both__ box_t &grow(box_t other)
    { lower = min(lower,other.lower); upper = max(upper,other.upper); return *this; }
    inline __both__ box_t &set_empty()
    {
      lower = make<vec_t>(empty_box_lower_value<scalar_t>());
      upper = make<vec_t>(empty_box_upper_value<scalar_t>());
      return *this;
    }

    inline __both__ box_t &grow(cuda_vec_t other)
    { lower = min(lower,make<vec_t>(other)); upper = max(upper,make<vec_t>(other)); return *this; }

    /*! for convenience's sake, get_lower(i) := lower[i] */
    inline __both__ scalar_t get_lower(int i) const { return lower[i]; }
    /*! for convenience's sake, get_upper(i) := upper[i] */
    inline __both__ scalar_t get_upper(int i) const { return upper[i]; }
      
    vec_t lower, upper;
  };

  using box3f = box_t<vec3f>;
}

