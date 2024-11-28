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

#define CUBQL_CPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/builder/cpu/spatialMedian.h"

namespace cuBQL {
  namespace cpu {
    
    // ==================================================================
    // for regular, BINARY BVHes
    // ==================================================================
    template<typename T, int D>
    void spatialMedian(BinaryBVH<T,D>   &bvh,
                       const box_t<T,D> *boxes,
                       uint32_t          numPrims,
                       BuildConfig       buildConfig);
    
    template<typename T, int D>
    inline void freeBVH(BinaryBVH<T,D> &bvh)
    {
      delete[] bvh.nodes;
      delete[] bvh.primIDs;
      bvh.nodes = 0;
      bvh.primIDs = 0;
    }

    // ==================================================================
    // for WIDE BVHes
    // ==================================================================
    template<typename T, int D, int WIDTH>
    void spatialMedian(WideBVH<T,D,WIDTH>   &bvh,
                       const box_t<T,D>     *boxes,
                       uint32_t              numPrims,
                       BuildConfig           buildConfig);
    
    template<typename T, int D, int WIDTH>
    inline void freeBVH(WideBVH<T,D,WIDTH> &bvh)
    {
      delete[] bvh.nodes;
      delete[] bvh.primIDs;
      bvh.nodes = 0;
      bvh.primIDs = 0;
    }
    
  }

  /*! non-specialized 'cuBQL::cpuBuilder' entry point purely witin the
    cuBQL:: namespace, which wraps all builder variants */
  template<typename T, int D, int WIDTH>
  void cpuBuilder(WideBVH<T,D,WIDTH>   &bvh,
                  const box_t<T,D>     *boxes,
                  uint32_t              numPrims,
                  BuildConfig           buildConfig)
  {
    /*! right now, only have a slow spatial median builder */
    cpu::spatialMedian(bvh,boxes,numPrims,buildConfig);
  }
  /*! non-specialized 'cuBQL::cpuBuilder' entry point purely witin the
    cuBQL:: namespace, which wraps all builder variants */
  template<typename T, int D>
  void cpuBuilder(BinaryBVH<T,D>   &bvh,
                  const box_t<T,D>     *boxes,
                  uint32_t              numPrims,
                  BuildConfig           buildConfig)
  {
    /*! right now, only have a slow spatial median builder */
    cpu::spatialMedian(bvh,boxes,numPrims,buildConfig);
  }
}
