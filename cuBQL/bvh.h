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

  struct box3f {
    enum { numDims = 3 };

    inline __device__ float get_lower(int d) const { return (&lower.x)[d]; }
    inline __device__ float get_upper(int d) const { return (&upper.x)[d]; }
  
    inline __device__ void set_empty() {
      lower = make_float3(+INFINITY,+INFINITY,+INFINITY);
      upper = make_float3(-INFINITY,-INFINITY,-INFINITY);
    }
    
    float3 lower, upper;
  };

  struct BinaryBVH {
    struct Node {
      box3f    bounds;
      uint64_t offset : 48;
      uint64_t count  : 16;
    };

    Node     *nodes;
    uint32_t  numNodes;
    uint32_t *primIDs;
    uint32_t  numPrims;
  };

  /*! builds a wide-bvh over a given set of primitmive bounding boxes.

    builder runs on the GPU; boxes[] must be a device-readable array
    (managed or device mem); bvh arrays will be allocated in device mem 

    primitives may be marked as "inactive/invalid" by using a bounding
    box whose lower/upper coordinates are inverted; such primitmives
    will be ignored, and will thus neither be visited during traversal
    nor mess up the tree in any way, shape, or form
  */
  void gpuBuilder(BinaryBVH &bvh,
                  const box3f *boxes,
                  uint32_t numBoxes,
                  int maxLeafSize);
  
  // template<typename prim_t, int BVH_WIDTH>
  // struct WideBVH {
  //   using box_t = typename prim_traits<prim_t>::box_t;
  
  //   struct Node {
  //     box_t bounds[BVH_WIDTH];
  //     struct {
  //       uint64_t offset : 48;
  //       uint64_t count  : 16;
  //     } child[BVH_WIDTH];
  //   };

  //   Node     *nodes;
  //   uint32_t  numNodes;
  //   uint32_t *primIDs;
  //   uint32_t  numPrims;
  // };
  
}

#if CUBQL_GPU_BUILDER_IMPLEMENTATION
# include "cuBQL/impl/gpu_builder.h"  
#endif


  
