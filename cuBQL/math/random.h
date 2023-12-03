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
#include <random>

namespace cuBQL {

  /*! simple 24-bit linear congruence generator */
  template<unsigned int N=4>
  struct LCG {
    
    inline __cubql_both LCG()
    { /* intentionally empty so we can use it in device vars that
         don't allow dynamic initialization (ie, PRD) */
    }
    inline __cubql_both LCG(unsigned int val0, unsigned int val1)
    { init(val0,val1); }

    inline __cubql_both void init(unsigned int val0, unsigned int val1)
    {
      unsigned int v0 = val0;
      unsigned int v1 = val1;
      unsigned int s0 = 0;
      
      for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
      }
      state = v0;
    }
    
    inline __cubql_both uint32_t ui32()
    {
      const uint32_t LCG_A = 1664525u;
      const uint32_t LCG_C = 1013904223u;
      state = (LCG_A * state + LCG_C);
      return uint32_t(state);
    }
    
    /*! Generate random unsigned int in [0, 2^24), then use that to
      generate random float in [0.f,1.f) */
    inline __cubql_both float operator() ()
    {
      const uint32_t LCG_A = 1664525u;
      const uint32_t LCG_C = 1013904223u;
      state = (LCG_A * state + LCG_C);
      return (state & 0x00FFFFFF) / (float) 0x01000000;
    }
    
    uint32_t state;
  };

}

