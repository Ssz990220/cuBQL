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
#include <cuBQL/math/vec.h>
#include <cuBQL/math/box.h>
#include <cuBQL/math/affine.h>
#include <nvfunctional>

namespace cuBQL {

  inline __cubql_both
  float sqrDistance(vec3f point, box3f box)
  {
    vec3f projected = min(max(point,box.lower),box.upper);
    vec3f v = projected - point;
    return dot(v,v);
  }

#if 0
  
  /*! performs a 'shrinking radius (leaf-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
  template<typename Lambda>
  inline __cubql_both
  void shrinkingRadiusQuery_forEachLeaf
  (bvh3f bvh,
   vec3f queryPoint,
   float sqrMaxSearchRadius,
   /*! lambda that gets called for each leaf that may contain any
     primitives. if this lamdba does find a new, better result than
     whatever the query had before this lambda MUST return the SQUARE
     of the new culling radius */
   // nvstd::function<float(const uint32_t*,uint64_t)> lambdaToExecuteForEachCandidateLeaf,
   const Lambda &lambdaToExecuteForEachCandidateLeaf,
   bool dbg = false)
  {
    float sqrCullDist = sqrMaxSearchRadius;
    struct StackEntry {
      union {
        struct {
          uint32_t idx;
          float    dist;
        };
        uint64_t forALign;
      };
    };
    const int stackSize = 64;
    StackEntry traversalStack[stackSize], *stackPtr = traversalStack;
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
        float d0 = sqrDistance(queryPoint,n0.bounds);
        float d1 = sqrDistance(queryPoint,n1.bounds);
        if (min(d0,d1) >= sqrCullDist) {
          // both children are too far away; this is a dead end
          node.count = 0;
          break;
        }

        uint32_t farID;
        if (d0 < d1) {
          // go left side, possibly pop right side
          node = n0.admin;
          farID = n1Idx;
        } else {
          // go left side, possibly pop right side
          node = n1.admin;
          farID = n0Idx;
        }
        float farChildDist = max(d0,d1);
        bool  bothInRange = (farChildDist < sqrCullDist);
        if (bothInRange) {
          if (stackPtr >= traversalStack+stackSize) {
            printf("stack overflow\n");
            return;
          }
          *stackPtr++ = StackEntry{ farID, farChildDist }; }
      }
      
      if (node.count != 0) {
        // we're at a valid leaf: call the lambda and see if that gave
        // us a enw, closer cull radius
        float leafResult
          = lambdaToExecuteForEachCandidateLeaf(bvh.primIDs+node.offset,node.count);
        if (leafResult < 0.f) return;
        sqrCullDist = min(sqrCullDist,leafResult);
      }
      // ------------------------------------------------------------------
      // pop next un-traversed node from stack, discarding any nodes
      // that are more distant than whatever query radius we now have
      // ------------------------------------------------------------------
      while (true) {
        if (stackPtr == traversalStack)
          return;
        StackEntry fromStack = *--stackPtr;
        if (fromStack.dist <= sqrCullDist) {
          node = bvh.nodes[fromStack.idx].admin;
          // found one!
          break;
        }
        // discard this one, it's too far away (apparently the search
        // radius has shrunk since we pushed this node
        continue;
      }
    }
  }
  /*! performs a 'shrinking radius (primitive-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
  template<typename Lambda>
  inline __cubql_both
  void shrinkingRadiusQuery_forEachPrim
  (/* the bvh we're querying into */
   bvh3f bvh,
   /*! the center of out query ball */
   vec3f queryPoint,
   /*! the SQUARE of the maximum query radius to which we want to
     restrict the search; can be INFINITY for unrestricted searches */
   float sqrMaxSearchRadius,
   /*! lambda that gets called for each candidate primitmive index
     that may contain any primitmives. if this lamdba does find a new,
     better result than whatever the query had before this lambda MUST
     return the SQUARE of the new culling radius */
   const Lambda &lambdaToExecuteForEachCandidate,
   // nvstd::function<float(uint32_t)> lambdaToExecuteForEachCandidate,
   bool dbg = false)
  {
    auto leafCode
      = [lambdaToExecuteForEachCandidate](const uint32_t *leafPrims,
                                          size_t numPrims)->float
      {
        float leafResult = INFINITY;
        for (int i=0;i<numPrims;i++) {
          float primResult
            = lambdaToExecuteForEachCandidate(leafPrims[i]);
          leafResult = min(leafResult,primResult);
        }
        return leafResult;
      };
    shrinkingRadiusQuery_forEachLeaf(bvh,queryPoint,sqrMaxSearchRadius,leafCode,dbg);
  }
  
  
#else
  
  /*! performs a 'shrinking radius (leaf-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
  inline __cubql_both
  void shrinkingRadiusQuery_forEachLeaf
  (bvh3f bvh,
   vec3f queryPoint,
   float sqrMaxSearchRadius,
   /*! lambda that gets called for each leaf that may contain any
     primitives. if this lamdba does find a new, better result than
     whatever the query had before this lambda MUST return the SQUARE
     of the new culling radius */
   const nvstd::function<float(const uint32_t*,uint64_t)> &lambdaToExecuteForEachCandidateLeaf,
   bool dbg = false)
  {
    float sqrCullDist = sqrMaxSearchRadius;
    struct StackEntry {
      union {
        struct {
          uint32_t idx;
          float    dist;
        };
        uint64_t forALign;
      };
    };
    const int stackSize = 64;
    StackEntry traversalStack[stackSize], *stackPtr = traversalStack;
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
        float d0 = sqrDistance(queryPoint,n0.bounds);
        float d1 = sqrDistance(queryPoint,n1.bounds);
        if (min(d0,d1) >= sqrCullDist) {
          // both children are too far away; this is a dead end
          node.count = 0;
          break;
        }

        uint32_t farID;
        if (d0 < d1) {
          // go left side, possibly pop right side
          node = n0.admin;
          farID = n1Idx;
        } else {
          // go left side, possibly pop right side
          node = n1.admin;
          farID = n0Idx;
        }
        float farChildDist = max(d0,d1);
        bool  bothInRange = (farChildDist < sqrCullDist);
        if (bothInRange) {
          if (stackPtr >= traversalStack+stackSize) {
            printf("stack overflow\n");
            return;
          }
          *stackPtr++ = StackEntry{ farID, farChildDist }; }
      }
      
      if (node.count != 0) {
        // we're at a valid leaf: call the lambda and see if that gave
        // us a enw, closer cull radius
        float leafResult
          = lambdaToExecuteForEachCandidateLeaf(bvh.primIDs+node.offset,node.count);
        if (leafResult < 0.f) return;
        sqrCullDist = min(sqrCullDist,leafResult);
      }
      // ------------------------------------------------------------------
      // pop next un-traversed node from stack, discarding any nodes
      // that are more distant than whatever query radius we now have
      // ------------------------------------------------------------------
      while (true) {
        if (stackPtr == traversalStack)
          return;
        StackEntry fromStack = *--stackPtr;
        if (fromStack.dist <= sqrCullDist) {
          node = bvh.nodes[fromStack.idx].admin;
          // found one!
          break;
        }
        // discard this one, it's too far away (apparently the search
        // radius has shrunk since we pushed this node
        continue;
      }
    }
  }
  /*! performs a 'shrinking radius (primitive-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
  inline __cubql_both
  void shrinkingRadiusQuery_forEachPrim
  (/* the bvh we're querying into */
   bvh3f bvh,
   /*! the center of out query ball */
   vec3f queryPoint,
   /*! the SQUARE of the maximum query radius to which we want to
     restrict the search; can be INFINITY for unrestricted searches */
   float sqrMaxSearchRadius,
   /*! lambda that gets called for each candidate primitmive index
     that may contain any primitmives. if this lamdba does find a new,
     better result than whatever the query had before this lambda MUST
     return the SQUARE of the new culling radius */
   // const Lambda &lambdaToExecuteForEachCandidate,
   const nvstd::function<float(uint32_t)> &lambdaToExecuteForEachCandidate,
   bool dbg = false)
  {
    auto leafCode
      = [&lambdaToExecuteForEachCandidate](const uint32_t *leafPrims,
                                           size_t numPrims)->float
      {
        float leafResult = INFINITY;
        for (int i=0;i<numPrims;i++) {
          float primResult
            = lambdaToExecuteForEachCandidate(leafPrims[i]);
          leafResult = min(leafResult,primResult);
        }
        return leafResult;
      };
    shrinkingRadiusQuery_forEachLeaf(bvh,queryPoint,sqrMaxSearchRadius,leafCode,dbg);
  }
  
  
#endif
  
}
 
