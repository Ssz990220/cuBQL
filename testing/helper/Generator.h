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

#include "testing/helper.h"
#include "testing/helper/CUDAArray.h"
#include <memory>

namespace testing {

  template<typename T, int D>
  struct PointGenerator {
    typedef std::shared_ptr<PointGenerator> SP;
    
    virtual CUDAArray<vec_t<T,D>> generate(int count, int seed);
  };
  
  template<typename T, int D>
  struct UniformPointGenerator : public PointGenerator<T, D>
  {
    virtual CUDAArray<vec_t<T,D>> generate(int count, int seed);
  };

  template<typename T, int D>
  struct ClusteredPointGenerator : public PointGenerator<T, D>
  {
    virtual CUDAArray<vec_t<T,D>> generate(int count, int seed);
  };

  template<typename T, int D>
  struct PointTranslator : public PointGenerator<T, D>
  {
    typename PointGenerator<T,D>::SP source;
    PointTranslator(typename PointGenerator<T,D>::SP source)
      : source(source)
    {}
    virtual CUDAArray<vec_t<T,D>> generate(int count, int seed);
  };
  
  
  template<typename T, int D>
  struct BoxGenerator {
    typedef std::shared_ptr<BoxGenerator<T,D>> SP;
    
    virtual CUDAArray<box_t<T,D>> generate(int count, int seed);
  };


  template<typename T, int D>
  struct PointsToBoxes : public BoxGenerator<T,D> {
    using vec_t = typename cuBQL::vec_t<T,D>;
    
    typename PointGenerator<T, D>::SP pointGenerator;

    PointsToBoxes(typename PointGenerator<T,D>::SP pointGenerator)
      : pointGenerator(pointGenerator)
    {}

    vec_t boxSize;
    
    virtual CUDAArray<box_t<T,D>> generate(int count, int seed);
  };

  template<typename T, int D>
  struct BoxMixture : public BoxGenerator<T,D> {
    virtual CUDAArray<box_t<T,D>> generate(int count, int seed);
    
    typename BoxGenerator<T,D>::SP gen_a;
    typename BoxGenerator<T,D>::SP gen_b;
    float prob_a;
  };


}


