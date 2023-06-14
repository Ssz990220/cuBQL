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

#include "cuBQL/common/common.h"
#include "cuBQL/common/vec.h"
#include "cuBQL/common/box.h"

namespace cuBQL {

  using ::min;
  using ::max;

  using ::make_float3;
  
  template<int N> struct log_of;
  template<> struct log_of< 2> { enum { value = 1 }; };
  template<> struct log_of< 4> { enum { value = 2 }; };
  template<> struct log_of< 8> { enum { value = 3 }; };
  template<> struct log_of<16> { enum { value = 4 }; };

  inline __both__ float sqr(float f) { return f*f; }
  inline __both__ float3 make_float3(float f) { return make_float3(f,f,f); }
    
  inline __both__ float3 min(float3 a, float3 b)
  { return make_float3(::min(a.x,b.x),::min(a.y,b.y),::min(a.z,b.z)); }
  
  inline __both__ float3 max(float3 a, float3 b)
  { return make_float3(::max(a.x,b.x),::max(a.y,b.y),::max(a.z,b.z)); }

  inline __both__ float3 operator-(float3 a, float3 b)
  { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
  inline __both__ float3 operator+(float3 a, float3 b)
  { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }

  inline __both__ float3 operator*(float3 a, float3 b)
  { return make_float3(a.x*b.x,a.y*b.y,a.z*b.z); }

  inline __both__ float3 operator*(float a, float3 b)
  { return make_float3(a,a,a)*b; }
  
  inline __both__ float dot(float3 a, float3 b)
  { return a.x*b.x+a.y*b.y+a.z*b.z; }
  
  inline __both__ float3 cross(float3 b, float3 c)
  { return make_float3(b.y*c.z-b.z*c.y,
                       b.z*c.x-b.x*c.z,
                       b.x*c.y-b.y*c.x); }
  
  inline __both__ float sqrLength(float3 a)
  { return dot(a,a); }

  inline __both__ float length(float3 a)
  { return sqrtf(sqrLength(a)); }
  
  inline __both__ float sqrDistance(float3 a, float3 b)
  { return sqrLength(a-b); }

  inline __both__ float get(float3 a, int d)
  { return (d==0)?a.x:((d==1)?a.y:a.z); }
  
  
  // struct box3f {
  //   enum { numDims = 3 };

  //   inline __both__ float get_lower(int d) const { return (d==0)?lower.x:((d==1)?lower.y:lower.z); }
  //   inline __both__ float get_upper(int d) const { return (d==0)?upper.x:((d==1)?upper.y:upper.z); }
  //   inline __both__ void set_empty() {
  //     lower = make_float3(+INFINITY,+INFINITY,+INFINITY);
  //     upper = make_float3(-INFINITY,-INFINITY,-INFINITY);
  //   }
  //   inline __both__ void grow(const float3 p)
  //   {
  //     lower = min(lower,p);
  //     upper = max(upper,p);
  //   }
  //   inline __both__ void grow(const box3f other)
  //   {
  //     lower = min(lower,other.lower);
  //     upper = max(upper,other.upper);
  //   }
  //   float3 lower, upper;
  // };

  /*! deprecated...*/
  inline __both__
  box3f make_box3f(vec3f lower, vec3f upper) { return {lower,upper}; }
  /*! deprecated...*/
  inline __both__
  box3f make_box3f(float3 lower, float3 upper) { return {make<vec3f>(lower),make<vec3f>(upper)}; }
                   
  inline __both__
  void grow(box3f &box, const vec3f &other) { box.grow(other); }
  
  inline __both__
  void grow(box3f &box, const box3f &other) { box.grow(other); }
    
  inline __both__
  float surfaceArea(box3f box)
  {
    float sx = box.upper[0]-box.lower[0];
    float sy = box.upper[1]-box.lower[1];
    float sz = box.upper[2]-box.lower[2];
    // float sx = box.get_upper(0)-box.get_lower(0);
    // float sy = box.get_upper(1)-box.get_lower(1);
    // float sz = box.get_upper(2)-box.get_lower(2);
    return sx*sy + sx*sz + sy*sz;
  }

  inline __both__
  float3 centerOf(box3f box)
  {
    return 0.5f * (box.lower + box.upper);
  }

}

