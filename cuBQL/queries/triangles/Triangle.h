// ======================================================================== //
// Copyright 2024-2024 Ingo Wald                                            //
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

/*! \file cuBQL/triangles/Triangle.h Defines a generic triangle type and
    some operations thereon, that various queries can then build on */

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"

namespace cuBQL {
  
  struct Triangle {
    /*! returns an axis aligned bounding box enclosing this triangle */
    inline __cubql_both box3f bounds() const;
    inline __cubql_both vec3f sample(float u, float v) const;
    
    vec3f a, b, c;
  };

  inline __cubql_both box3f Triangle::bounds() const
  { return box3f().including(a).including(b).including(c); }

  inline __cubql_both float area(Triangle tri)
  { return length(cross(tri.b-tri.a,tri.c-tri.a)); }

  inline __cubql_both vec3f Triangle::sample(float u, float v) const
  {
    if (u+v >= 1.f) { u = 1.f-u; v = 1.f-v; }
    return (1.f-u-v)*a + u * b + v * c;
  }
    

} // cuBQL

