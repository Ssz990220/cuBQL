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

#include "cuBQL/math/vec.h"

namespace cuBQL {

  /*! @{ defines what box.lower/box.upper coordinate values to use for
      indicating an "empty box" of the given type */
  template<typename scalar_t>
  inline __both__ scalar_t empty_box_lower_value();
  template<typename scalar_t>
  inline __both__ scalar_t empty_box_upper_value();

  template<> inline __both__ float empty_box_lower_value<float>() { return +INFINITY; }
  template<> inline __both__ float empty_box_upper_value<float>() { return -INFINITY; }
  /*! @} */
  
  /*! a axis-aligned bounding box, made up of a 'lower' and 'upper'
      vector bounding it. any box with any lower[k] > upper[k] is to
      be treated as a "empty box" (e.g., for a bvh that might indicate
      a primitive that we want to ignore from the build) */
  template<typename _scalar_t, int _numDims>
  struct box_t {
    enum { numDims = _numDims };
    using scalar_t = _scalar_t;
    using vec_t = cuBQL::vec_t<scalar_t,numDims>;

    using cuda_vec_t = typename cuda_eq_t<scalar_t,numDims>::type;
    
    inline __both__ box_t &grow(const vec_t &v)
    { lower = min(lower,v); upper = max(upper,v); return *this; }
    inline __both__ box_t &grow(const box_t &other)
    { lower = min(lower,other.lower); upper = max(upper,other.upper); return *this; }
    inline __both__ box_t &set_empty()
    {
      lower = make<vec_t>(empty_box_lower_value<scalar_t>());
      upper = make<vec_t>(empty_box_upper_value<scalar_t>());
      return *this;
    }

    inline __both__ box_t &grow(cuda_vec_t other)
    {
      lower = min(lower,make<vec_t>(other));
      upper = max(upper,make<vec_t>(other)); return *this;
    }

    /*! returns the center of the box, up to rounding errors. (i.e. on
        its, the center of a box with lower=2 and upper=3 is 2, not
        2.5! */
    inline __both__ vec_t center() const
    { return (lower+upper)/scalar_t(2); }
    
    /*! returns TWICE the center (which happens to be the SUM of lower
        an dupper). Note this 'conceptually' the same as 'center()',
        but without the nasty division that may lead to rounding
        errors for int types; it's obviously not the center but twice
        the center - but as long as all routines that expect centers
        use that same 'times 2' this will still work out */
    inline __both__ vec_t twice_center() const
    { return (lower+upper); }

    /*! for convenience's sake, get_lower(i) := lower[i] */
    inline __both__ scalar_t get_lower(int i) const { return lower[i]; }
    /*! for convenience's sake, get_upper(i) := upper[i] */
    inline __both__ scalar_t get_upper(int i) const { return upper[i]; }
      
    vec_t lower, upper;
  };

  using box3f = box_t<float,3>;

  inline __both__
  float surfaceArea(box3f box)
  {
    const float sx = box.upper[0]-box.lower[0];
    const float sy = box.upper[1]-box.lower[1];
    const float sz = box.upper[2]-box.lower[2];
    return sx*sy + sx*sz + sy*sz;
  }

  
  template<typename T, int D> inline __both__
  typename dot_result_t<T>::type sqrDistance(box_t<T,D> box, vec_t<T,D> point)
  {
    vec_t<T,D> closestPoint = min(max(point,box.lower),box.upper);
    return sqrDistance(closestPoint,point);
  }

  template<typename T, int D> inline __both__
  float fSqrDistance(box_t<T,D> box, vec_t<T,D> point)
  {
    vec_t<T,D> closestPoint = min(max(point,box.lower),box.upper);
    return sqrDistance(closestPoint,point);
  }

  template<typename T, int D> inline __both__
  box_t<T,D> &grow(box_t<T,D> &b, vec_t<T,D> v)
  { b.grow(v); return b; }
  
  template<typename T, int D> inline __both__
  box_t<T,D> &grow(box_t<T,D> &b, box_t<T,D> ob)
  { b.grow(ob); return b; }




  template<typename T, int D> inline __both__
  box_t<T,D> make_box(vec_t<T,D> v) { return {v,v}; }
  
  
  // inline __both__
  // box3f make_box3f(vec3f lower, vec3f upper) { return {lower,upper}; }
  // /*! deprecated...*/
  // inline __both__
  // box3f make_box3f(float3 lower, float3 upper) { return {make<vec3f>(lower),make<vec3f>(upper)}; }
                   
  // inline __both__
  // void grow(box3f &box, const vec3f &other) { box.grow(other); }
  
  // inline __both__
  // void grow(box3f &box, const box3f &other) { box.grow(other); }
    
  // inline __both__
  // float3 centerOf(box3f box)
  // {
  //   return 0.5f * (box.lower + box.upper);
  // }

}

