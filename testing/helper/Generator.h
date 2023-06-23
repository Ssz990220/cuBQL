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

  /*! a 'point generator' is a class that implements a procedure to
      create a randomized set of points (of given type and
      dimensoins). In particular, this library allows for describing
      various point generators through a string, such as "uniform"
      (uniformly distributed points), "nrooks" (a n-rooks style
      distribution, see below), "clustered", etc. */
  template<typename T, int D>
  struct PointGenerator {
    typedef std::shared_ptr<PointGenerator> SP;

    static SP createFromString(const std::string &wholeString);
    
    static SP createAndParse(const char *&curr);
    virtual void parse(const char *&currentParsePos);
    
    virtual CUDAArray<vec_t<T,D>> generate(int count, int seed) = 0;
  };
  
  template<typename T, int D>
  struct UniformPointGenerator : public PointGenerator<T, D>
  {
    CUDAArray<vec_t<T,D>> generate(int count, int seed) override;
  };

  /*! re-maps points from the 'default domain' to the domain specified
      by [lower,upper]. Ie, for float the defulat domain is [0,1]^N,
      so a poitn with coordinate x=1 would be re-mapped to
      x=lower. Note this does not require the input points to *be*
      inside that default domain - if they are outside the domain the
      generated points will simply be outside the target domain,
      too 

      To create via generator string, use the string "remap x0 y0
      ... x1 y1 ... <source>", where x0,x1 etc are the lower coordinates of
      the target domain; x1, y1 etc are the upper bounds of the target
      domain, and <source> is another generator that produces the input
      points. E.g., assuimng we'd be dealing with <int,2> data, the string
      "remap 2 2 4 4 nrooks" would first generate points with the "nrooks"
      generator, then re-map those to [(2,2),(4,4)].
  */
  template<typename T, int D>
  struct RemapPointGenerator : public PointGenerator<T, D>
  {
    RemapPointGenerator();
    
    CUDAArray<vec_t<T,D>> generate(int count, int seed) override;
    
    virtual void parse(const char *&currentParsePos);

    vec_t<T,D> lower, upper;
    typename PointGenerator<T,D>::SP source;
  };

  template<typename T, int D>
  struct ClusteredPointGenerator : public PointGenerator<T, D>
  {
    CUDAArray<vec_t<T,D>> generate(int count, int seed) override;
    
    /*! num clusters to generate - if 0, we'll use D-th root of count */
    // int numClusters = 0;
  };

  template<typename T, int D>
  struct NRooksPointGenerator : public PointGenerator<T, D>
  {
    CUDAArray<vec_t<T,D>> generate(int count, int seed) override;
    
    /*! num clusters to generate - if 0, we'll use D-th root of count */
    // int numClusters = 0;
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


