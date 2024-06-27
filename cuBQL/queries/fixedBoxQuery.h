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

#include "cuBQL/bvh.h"
#include <nvfunctional>

namespace cuBQL {

#define CUBQL_TERMINATE_TRAVERSAL 1
#define CUBQL_CONTINUE_TRAVERSAL  0
  
  /*! This query finds all primitives within a given fixed (ie, never
      changing) axis-aligned cartesian box, and calls the provided
      callback-lambda for each such prim. The provided lambda can do
      with the provided prim as it pleases, and is to report either
      CUBQL_TERMINATE_TRAVERSAL (in which case traversal will
      immediately terminate), or CUBQL_CONTINUE_TRAVERSAL (in which
      case traversal will continue to the next respective primitmive
      within the box, if such exists. 
    
      for this "for each prim' variant, the lambda should have a signature of
      [](int primID)->int
  */
  template<typename T, int D>
  inline __device__
  void fixedBoxQuery_forEachPrim(const BinaryBVH<T,D>,
                                 const box3f queryBox,
                                 nvstd::function<int(uint32_t)> lambdaToCallOnEachPrim);
  
  template<typename T, int D>
  inline __device__
  void fixedBoxQuery_forEachLeaf(const BinaryBVH<T,D>,
                                 const box3f queryBox,
                                 nvstd::function<int(const uint32_t *, size_t)> lambdaToCallOnEachPrim);
  



  // ==================================================================
  // IMPLEMENTATION
  // ==================================================================

  template<typename T, int D>
  inline __device__
  void fixedBoxQuery_forEachLeaf(const BinaryBVH<T,D> bvh,
                                 const box3f queryBox,
                                 nvstd::function<int(const uint32_t *, size_t)> lambdaToCallOnEachLeaf)
  {
    struct StackEntry {
      uint32_t idx;
    };
    bvh3f::node_t::Admin traversalStack[64], *stackPtr = traversalStack;
    bvh3f::node_t::Admin node = bvh.nodes[0].admin;
    // ------------------------------------------------------------------
    // traverse until there's nothing left to traverse:
    // ------------------------------------------------------------------
    while (true) {

      // ------------------------------------------------------------------
      // traverse INNER nodes downward; breaking out if we either find
      // a leaf within the current search radius, or found a dead-end
      // at which we need to pop
      // ------------------------------------------------------------------
      while (true) {
        if (node.count != 0)
          // it's a boy! - seriously: this is not a inner node, step
          // out of down-travesal and let leaf code pop in.
          break;

        uint32_t n0Idx = node.offset+0;
        uint32_t n1Idx = node.offset+1;
        bvh3f::node_t n0 = bvh.nodes[n0Idx];
        bvh3f::node_t n1 = bvh.nodes[n1Idx];
        bool o0 = queryBox.overlaps(n0.bounds);
        bool o1 = queryBox.overlaps(n1.bounds);
        if (o0) {
          if (o1) {
            *stackPtr++ = n1.admin;
          } else {
          }
          node = n0.admin;
        } else {
          if (o1) {
            node = n1.admin;
          } else {
            // both children are too far away; this is a dead end
            node.count = 0;
            break;
          }
        }
      }
      
      if (node.count != 0) {
        // we're at a valid leaf: call the lambda and see if that gave
        // us a enw, closer cull radius
        int leafResult
          = lambdaToCallOnEachLeaf(bvh.primIDs+node.offset,node.count);
        if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
          return;
      }
      // ------------------------------------------------------------------
      // pop next un-traversed node from stack, discarding any nodes
      // that are more distant than whatever query radius we now have
      // ------------------------------------------------------------------
      if (stackPtr == traversalStack)
        return;
      node = *--stackPtr;
    }
  }

  template<typename T, int D>
  inline __device__
  void fixedBoxQuery_forEachPrim(const BinaryBVH<T,D> bvh,
                                 const box3f queryBox,
                                 nvstd::function<int(uint32_t)> lambdaToCallOnEachPrim)
  {
    auto leafCode = [lambdaToCallOnEachPrim](const uint32_t *primIDs, size_t numPrims) -> int
    {
      for (int i=0;i<(int)numPrims;i++)
        if (lambdaToCallOnEachPrim(primIDs[i]) == CUBQL_TERMINATE_TRAVERSAL)
          return CUBQL_TERMINATE_TRAVERSAL;
      return CUBQL_CONTINUE_TRAVERSAL;
    };
    
  }
}
