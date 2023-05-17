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

#include "cuBQL/math.h"
#include "cuBQL/bvh.h"

namespace cuBQL {

  inline __device__
  float3 project(box3f box, float3 point)
  { return max(min(point,box.upper),box.lower); }
  
  inline __device__
  float sqrDistance(box3f box, float3 point)
  { return sqrDistance(project(box,point),point); }

  inline __device__
  float sqrDistance(BinaryBVH::Node node, float3 point)
  { return sqrDistance(node.bounds,point); }
  
  inline __device__
  int fcp(BinaryBVH bvh,
          const float3 *dataPoints,
          float3 query,
          float maxQueryDist = INFINITY)
  {
    float cullDist = maxQueryDist*maxQueryDist;
    int result = -1;
    
    int2 stackBase[32], *stackPtr = stackBase;
    int nodeID = 0;
    int offset = 0;
    int count  = 0;
    while (true) {
      while (true) {
        // printf("nodeID %i\n",nodeID);
        offset = bvh.nodes[nodeID].offset;
        count  = bvh.nodes[nodeID].count;
        if (count>0)
          // leaf
          break;
        BinaryBVH::Node child0 = bvh.nodes[offset+0];
        BinaryBVH::Node child1 = bvh.nodes[offset+1];
        // printf(" child0 (%f %f %f)(%f %f %f)\n",
        //        child0.bounds.lower.x,
        //        child0.bounds.lower.y,
        //        child0.bounds.lower.z,
        //        child0.bounds.upper.x,
        //        child0.bounds.upper.y,
        //        child0.bounds.upper.z);
        // printf(" child1 (%f %f %f)(%f %f %f)\n",
        //        child1.bounds.lower.x,
        //        child1.bounds.lower.y,
        //        child1.bounds.lower.z,
        //        child1.bounds.upper.x,
        //        child1.bounds.upper.y,
        //        child1.bounds.upper.z);
        float dist0 = sqrDistance(child0,query);
        float dist1 = sqrDistance(child1,query);
        // printf("distances %f %f (vs %f)\n",dist0,dist1,cullDist);
        int closeChild = offset + ((dist0 > dist1) ? 1 : 0);
        if (dist1 <= cullDist) {
          float dist = max(dist0,dist1);
          int distBits = __float_as_int(dist);
          *stackPtr++ = make_int2(closeChild^1,distBits);
        }
        if (min(dist0,dist1) > cullDist) {
          count = 0;
          break;
        }
        nodeID = closeChild;
      }
      // printf("--- at leaf %i %i\n",offset,count);
      for (int i=0;i<count;i++) {
        int primID = bvh.primIDs[offset+i];
        float dist = sqrDistance(dataPoints[primID],query);
        // printf(" > prim %i dist %f\n",primID,dist);
        if (dist >= cullDist) continue;
        cullDist = dist;
        result = primID;
      }
      while (true) {
        if (stackPtr == stackBase)
          return result;
        --stackPtr;
        if (__int_as_float(stackPtr->y) > cullDist) continue;
        nodeID = stackPtr->x;
        break;
      }
    }
  }
  
}

