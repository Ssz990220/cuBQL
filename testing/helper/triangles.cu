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

#include "cuBQL/math/random.h"
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

namespace cuBQL {
  namespace test_rig {

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
      const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
      for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
        tinyobj::shape_t &shape = shapes[shapeID];
        for (size_t faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
          tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
          tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
          tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
        
          vec3f a = vertex_array[idx0.vertex_index];
          vec3f b = vertex_array[idx1.vertex_index];
          vec3f c = vertex_array[idx2.vertex_index];
          triangles.push_back({a,b,c});
        }
      }
      return triangles;
    }
  
    void saveOBJ(const std::vector<Triangle> &triangles,
                 const std::string &outFileName)
    {
      std::ofstream out(outFileName.c_str());
      for (auto tri : triangles) {
        out << "v " << tri.a.x << " " << tri.a.y << " " << tri.a.z << std::endl;
        out << "v " << tri.b.x << " " << tri.b.y << " " << tri.b.z << std::endl;
        out << "v " << tri.c.x << " " << tri.c.y << " " << tri.c.z << std::endl;
        out << "f -1 -2 -3" << std::endl;
      }
    }
  
    std::vector<Triangle> triangulate(const std::vector<box3f> &boxes)
    {
      std::vector<Triangle> triangles;
      int indices[] = {0,1,3, 2,3,0,
                       5,7,6, 5,6,4,
                       0,4,5, 0,5,1,
                       2,3,7, 2,7,6,
                       1,5,7, 1,7,3,
                       4,0,2, 4,2,6};
      Triangle tri;
      for (auto box : boxes) {
        vec3f vertices[8], *vtx = vertices;
        for (int iz=0;iz<2;iz++)
          for (int iy=0;iy<2;iy++)
            for (int ix=0;ix<2;ix++) {
              vtx->x = (ix?box.lower:box.upper).x;
              vtx->y = (iy?box.lower:box.upper).y;
              vtx->z = (iz?box.lower:box.upper).z;
              vtx++;
            }
        for (int i=0;i<12;i++) {
          tri.a = vertices[indices[3*i+0]];
          tri.b = vertices[indices[3*i+1]];
          tri.c = vertices[indices[3*i+2]];
          triangles.push_back(tri);
        }
      }
      return triangles;
    }
  
    std::vector<vec3f> sample(const std::vector<Triangle> &triangles,
                              size_t numSamples,
                              int seed)
    {
      std::vector<float> cdf;
      float sum = 0.f;
      for (auto tri : triangles) {
        float a = area(tri);
        sum += a;
        cdf.push_back(sum);
      }
      for (auto &f : cdf)
        f *= 1./sum;
      cdf.back() = 1.f;

      std::default_random_engine reng;
      reng.seed(seed);
      std::uniform_real_distribution<float> rnd(0.f,1.f);

      std::vector<vec3f> points;
      for (int i=0;i<numSamples;i++) {
        int idx
          = (int)min(std::lower_bound(cdf.begin(),cdf.end(),rnd(reng)) - cdf.begin(),
                     cdf.size()-1);
        
        Triangle tri = triangles[idx];
        float u = rnd(reng);
        float v = rnd(reng);
        if (u+v > 1.f) { u = 1.f-u; v = 1.f-v; };
        points.push_back((1.f-u-v)*tri.a + u*tri.b + v*tri.c);
      }
      return points;
    }
  
  } // ::cuBQL::test_rig
} // ::cuBQL

