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

#include "cuBQL/common.h"

namespace cuBQL {

  using ::min;
  using ::max;
  
  inline __device__ float3 min(float3 a, float3 b)
  { return make_float3(::min(a.x,b.x),::min(a.y,b.y),::min(a.z,b.z)); }
  
  inline __device__ float3 max(float3 a, float3 b)
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
  
  
}

