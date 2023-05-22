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

#include "cuBQL/common/math.h"

namespace cuBQL {

  /*! struct used to control how exactly the builder is supposed to
      build the tree; in particular, at which threshold to make a
      leaf */
  struct BuildConfig {
    inline BuildConfig &enableSAH() { buildMethod = SAH; return *this; }
    typedef enum
      {
       /*! simple 'adaptive spatial median' strategy. When splitting a
         subtree, this first computes the centroid of each input
         primitive in that subtree, then computes the bounding box of
         those centroids, then creates a split plane along the widest
         dimension of that centroid boundig box, right through th
         emiddle */
       SPATIAL_MEDIAN=0,
       /*! use good old surface area heurstic. In theory that only
         makes sense for BVHes that are used for tracing rays
         (theoretic motivation is a bit wobbly for other sorts of
         queries), but it seems to help even for other queries. Much
         more expensive to build, though */
       SAH
    } BuildMethod;
    
    /*! what leaf size the builder is _allowed_ to make; no matter
        what input is specified, the builder may never produce leaves
        larger thn this value */
    int maxAllowedLeafSize = 1<<15;

    /*! threshold below which the builder should make a leaf, no
        matter what the prims in the subtree look like. A value of 0
        means "leave it to the builder" */
    int makeLeafThreshold = 0;

    BuildMethod buildMethod = SPATIAL_MEDIAN;
  };

  /*! the most basic type of BVH where each BVH::Node is either a leaf
      (and contains Node::count primitmives), or is a inner node (and
      points to a pair of child nodes). Node 0 is the root node; node
      1 is always unused (so all other node pairs start on n even
      index) */
  struct BinaryBVH {
    struct CUBQL_ALIGN(16) Node {
      box3f    bounds;
      /*! For inner nodes, this points into the nodes[] array, with
          left child at nodes.offset+0, and right chlid at
          nodes.offset+1. For leaf nodes, this points into the
          primIDs[] array, which first prim beign primIDs[offset],
          next one primIDs[offset+1], etc. */
      uint64_t offset : 48;
      /* number of primitives in this leaf, if a leaf; 0 for inner
         nodes. */
      uint64_t count  : 16;
    };

    Node     *nodes    = 0;
    uint32_t  numNodes = 0;
    uint32_t *primIDs  = 0;
    uint32_t  numPrims = 0;
  };

  /*! a 'wide' BVH in which each node has a fixed number of
    `BVH_WIDTH` children (some of those children can be un-used) */
  template<int BVH_WIDTH>
  struct WideBVH {

    /*! a n-wide node of this BVH; note that unlike BinaryBVH::Node
      this is not a "single" node, but actually N nodes merged
      together */
    struct CUBQL_ALIGN(16) Node {
      struct Child {
        box3f    bounds;
        struct {
          uint64_t valid  :  1;
          uint64_t offset : 45;
          uint64_t count  : 16;
        };
      } children[BVH_WIDTH];
    };

    Node     *nodes    = 0;
    //! number of (multi-)nodes on this WideBVH
    uint32_t  numNodes = 0;
    uint32_t *primIDs  = 0;
    uint32_t  numPrims = 0;
  };

  // ------------------------------------------------------------------
  
  /*! builds a wide-bvh over a given set of primitive bounding boxes.

    builder runs on the GPU; boxes[] must be a device-readable array
    (managed or device mem); bvh arrays will be allocated in device mem 

    input primitives may be marked as "inactive/invalid" by using a
    bounding box whose lower/upper coordinates are inverted; such
    primitmives will be ignored, and will thus neither be visited
    during traversal nor mess up the tree in any way, shape, or form
  */
  void gpuBuilder(BinaryBVH   &bvh,
                  const box3f *boxes,
                  uint32_t     numBoxes,
                  BuildConfig  buildConfig,
                  cudaStream_t s=0);
  
  /*! builds a BinaryBVH over the given set of boxes (using the given
      stream), using a simple adaptive spatial median builder (ie,
      each subtree will be split by first computing the bounding box
      of all its contained primitives' spatial centers, then choosing
      a split plane that splits this cntroid bounds in the center,
      along the widest dimension. Leaves will be created once the size
      of a subtree get to or below buildConfig.makeLeafThreshold */
  template<int N>
  void gpuBuilder(WideBVH<N>  &bvh,
                  const box3f *boxes,
                  uint32_t     numBoxes,
                  BuildConfig  buildConfig,
                  cudaStream_t s=0);

  // ------------------------------------------------------------------
  
  /*! frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
      building the BVH. this assumes */
  void free(BinaryBVH   &bvh,
            cudaStream_t s=0);

  /*! frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
      building the BVH. this assumes */
  template<int N>
  void free(WideBVH<N>  &bvh,
            cudaStream_t s=0);

  // ------------------------------------------------------------------
  
  /*! computes the SAH cost of a already built BinaryBVH. This is
      often a useful metric for how "good" a BVH is */
  float computeSAH(const BinaryBVH &bvh);
  
  /*! computes the SAH cost of a already built WideBVH. This is often a
      useful metric for how "good" a BVH is */
  template<int N>
  float computeSAH(const WideBVH<N> &bvh);
} // ::cuBQL

#if CUBQL_GPU_BUILDER_IMPLEMENTATION
# include "cuBQL/impl/gpu_builder.h"  
#endif
#if CUBQL_GPU_BUILDER_IMPLEMENTATION
# include "cuBQL/impl/sah_builder.h"  
#endif
#if CUBQL_GPU_BUILDER_IMPLEMENTATION
# include "cuBQL/impl/wide_gpu_builder.h"  
#endif



  
