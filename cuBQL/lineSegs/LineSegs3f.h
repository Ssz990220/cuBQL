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

namespace cuBQL {
  namespace lineSegs {

    /*! a line segment, referring to two input points that define the
        two opposite corners of that line segment */
    struct IndexedSegment {
      int begin, end;
    };

    /* an actual line segment with actual points */
    struct Segment {
      vec3f begin, end;
    };

    /*! result of a fcp (find closest point) query */
    struct FCPResult {
      inline __device__ void clear(float maxDistSqr) { primID = -1; sqrDistance = maxDistSqr; }
      
      int   primID;
      float u;
      float sqrDistance;
    };
      
    inline __device__
    void fcp(FCPResult &result,
             const bvh3f                       bvh,
             const Segment *const __restrict__ segments);

    /*! find closest point (to query point) among a set of line
        segments (given by segments[] and vertices[], up to a maximum
        (square) query distance provided in result.sqrDistance. any
        line egments further away than result.sqrDistance will get
        rejected; at the end of the query result.maxSqrDistance will
        be the (square) distnace to the found segment (if found), or
        will be left un-modified if no such segment could be found
        within the initial query radius */
    inline __device__
    void fcp(FCPResult &result,
             const vec3f                              queryPoint,
             const bvh3f                              bvh,
             const IndexedSegment *const __restrict__ segments,
             const vec3f          *const __restrict__ vertices);

    
    // ==================================================================
    // implementation
    // ==================================================================
    
    /*! result of a closest-point intersection operation */
    struct CPResult {
      vec3f point;
      
      /*! parameterized distance along the line segment of where the
          hit is; u=0 being begin point, u=1 being end point */
      float u;

      float sqrDistance;
    };
    
    /*! compute point on 'segment' that is closest to 'queryPoint',
      and return the square distance to that point. */
    inline __device__
    CPResult closestPoint(const vec3f queryPoint, const Segment segment);
    
    

    /*! compute point on 'segment' that is closest to 'queryPoint',
      and return the square distance to that point. */
    inline __device__
    CPResult closestPoint(const vec3f queryPoint, const Segment segment)
    {
      CPResult result;
      vec3f ab = segment.end - segment.begin;
      float sqrLenAB = dot(ab,ab);
      if (sqrLenAB == 0.f) {
        result.u = 0.f;
      } else {
        result.u = dot(queryPoint - segment.begin,ab) / sqrtf(sqrLenAB);
      }
      result.point = segment.begin + result.u * ab;
      result.sqrDistance = sqrDistance(queryPoint,result.point);
      return result;
    }
    
    /*! find closest point (to query point) among a set of line
        segments (given by segments[] and vertices[], up to a maximum
        (square) query distance provided in result.sqrDistance. any
        line egments further away than result.sqrDistance will get
        rejected; at the end of the query result.maxSqrDistance will
        be the (square) distnace to the found segment (if found), or
        will be left un-modified if no such segment could be found
        within the initial query radius */
    inline __device__
    void fcp(FCPResult &result,
             const vec3f                              queryPoint,
             const bvh3f                              bvh,
             const IndexedSegment *const __restrict__ segments,
             const vec3f          *const __restrict__ vertices)
    {
      using node_t = typename bvh3f::Node;

      int2 stackBase[32], *stackPtr = stackBase;
      int nodeID = 0;
      int offset = 0;
      int count  = 0;
      while (true) {
        while (true) {
          offset = bvh.nodes[nodeID].admin.offset;
          count  = bvh.nodes[nodeID].admin.count;
          if (count>0)
            // leaf
            break;
          const node_t child0 = bvh.nodes[offset+0];
          const node_t child1 = bvh.nodes[offset+1];
          float dist0 = fSqrDistance(child0.bounds,queryPoint);
          float dist1 = fSqrDistance(child1.bounds,queryPoint);
          int closeChild = offset + ((dist0 > dist1) ? 1 : 0);
          if (dist1 < result.sqrDistance) {
            float dist = max(dist0,dist1);
            int distBits = __float_as_int(dist);
            *stackPtr++ = make_int2(closeChild^1,distBits);
          }
          if (min(dist0,dist1) > result.sqrDistance) {
            count = 0;
            break;
          }
          nodeID = closeChild;
        }
        for (int i=0;i<count;i++) {
          int primID = bvh.primIDs[offset+i];
          // if (primID == primIDtoIgnore) continue;

          IndexedSegment indices=segments[primID];
          Segment seg{vertices[indices.begin],vertices[indices.end]};
          CPResult primResult = closestPoint(queryPoint,seg);
          if (primResult.sqrDistance < result.sqrDistance) {
            result.primID = primID;
            result.u = primResult.u;
            result.sqrDistance = primResult.sqrDistance;
          }
        }
        while (true) {
          if (stackPtr == stackBase) 
            return;
          --stackPtr;
          if (__int_as_float(stackPtr->y) > result.sqrDistance) continue;
          nodeID = stackPtr->x;
          break;
        }
      }
    }
  }
}
