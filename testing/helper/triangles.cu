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

#include "testing/helper/triangles.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

namespace testing {

  std::vector<Triangle> loadOBJ(const std::string &objFile)
  {
    std::string modelDir = "";
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    
    std::string err = "";
    bool readOK
      = tinyobj::LoadObj(&attributes,
                         &shapes,
                         &materials,
                         &err,
                         &err,
                         objFile.c_str(),
                         modelDir.c_str(),
                         /* triangulate */true);
    if (!readOK) 
      throw std::runtime_error("Could not read OBJ model from "+objFile+" : "+err);

    std::vector<Triangle> triangles;
    const float3 *vertex_array   = (const float3*)attributes.vertices.data();
    for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
      tinyobj::shape_t &shape = shapes[shapeID];
      for (size_t faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
        tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
        tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
        tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
        
        float3 a = vertex_array[idx0.vertex_index];
        float3 b = vertex_array[idx1.vertex_index];
        float3 c = vertex_array[idx2.vertex_index];
        triangles.push_back({a,b,c});
      }
    }
    return triangles;
  }
  
  std::vector<float3> sample(const std::vector<Triangle> &triangles, size_t numSamples)
  {
    std::vector<float> cdf;
    float sum = 0.f;
    for (auto tri : triangles) {
      float3 n = cross(tri.b-tri.a,tri.c-tri.a);
      float area = length(n);
      sum += area;
      cdf.push_back(sum);
    }
    for (auto &f : cdf)
      f *= 1./sum;
    cdf.back() = 1.f;

    std::vector<float3> points;
    for (int i=0;i<numSamples;i++) {
      float rnd = drand48();
      int idx = std::lower_bound(cdf.begin(),cdf.end(),rnd) - cdf.begin();
      Triangle tri = triangles[idx];
      float u = drand48();
      float v = drand48();
      if (u+v > 1.f) { u = 1.f-u; v = 1.f - v; };
      points.push_back((1.f-u-v)*tri.a + u*tri.b + v*tri.c);
    }
    return points;
  }
  
}

