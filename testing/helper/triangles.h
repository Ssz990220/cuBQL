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
#include "testing/helper.h"

namespace cuBQL {
  namespace test_rig {

    struct Triangle {
      vec3f a, b, c;
    };

    inline __cubql_both float area(Triangle tri)
    { return length(cross(tri.b-tri.a,tri.c-tri.a)); }

    std::vector<Triangle> loadOBJ(const std::string &fileName);
    void loadOBJ(std::vector<vec3i> &indices,
                 std::vector<vec3f> &vertices,
                 const std::string &fileName);
    std::vector<Triangle> triangulate(const std::vector<box3f> &boxes);

    std::vector<vec3f> sample(const std::vector<Triangle> &triangles,
                              size_t numSamples,
                              int seed=0x34234987);
    void saveOBJ(const std::vector<Triangle> &triangles, const std::string &fileName);

  } // ::cuBQL::test_rig
} // ::cuBQL
