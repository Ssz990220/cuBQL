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

#if DO_STATS
# define STATS(a) a

  struct Stats {
    unsigned long long numNodes;
    unsigned long long numPrims;
  };
#else
# define STATS(a) 
#endif
  
namespace cuBQL {

  template<int D>
  inline __device__
  int fcp(const BinaryBVH<float,D> bvh,
#if USE_BOXES
          const box_t<float,D>    *prims,
#else
          const vec_t<float,D>    *prims,
#endif
          const vec_t<float,D>     query,
          /* in: SQUARE of max search distance; out: sqrDist of closest point */
          float          &maxQueryDistSquare
#if DO_STATS
          , Stats *d_stats = 0
#endif
          )
  {
    int result = -1;
    
    int2 stackBase[32], *stackPtr = stackBase;
    int nodeID = 0;
    int offset = 0;
    int count  = 0;
#if DO_STATS
    int numNodes = 0, numPrims = 0;
#endif
    while (true) {
      while (true) {
        offset = bvh.nodes[nodeID].admin.offset;
        count  = bvh.nodes[nodeID].admin.count;
#if DO_STATS
        numNodes++;
#endif
        if (count>0)
          // leaf
          break;
        typename BinaryBVH<float,D>::Node child0 = bvh.nodes[offset+0];
        typename BinaryBVH<float,D>::Node child1 = bvh.nodes[offset+1];
        float dist0 = fSqrDistance(child0.bounds,query);
        float dist1 = fSqrDistance(child1.bounds,query);
        int closeChild = offset + ((dist0 > dist1) ? 1 : 0);
        if (dist1 < maxQueryDistSquare) {
          float dist = max(dist0,dist1);
          int distBits = __float_as_int(dist);
          *stackPtr++ = make_int2(closeChild^1,distBits);
        }
        if (min(dist0,dist1) > maxQueryDistSquare) {
          count = 0;
          break;
        }
        nodeID = closeChild;
      }
      for (int i=0;i<count;i++) {
        int primID = bvh.primIDs[offset+i];
#if DO_STATS
        numPrims++;
#endif
        float dist2 = sqrDistance(prims[primID],query);
        if (dist2 >= maxQueryDistSquare) continue;
        maxQueryDistSquare = dist2;
        result             = primID;
      }
      while (true) {
        if (stackPtr == stackBase) {
#if DO_STATS
          if (d_stats) {
            atomicAdd(&d_stats->numNodes,(unsigned long long)numNodes);
            atomicAdd(&d_stats->numPrims,(unsigned long long)numPrims);
          }
#endif
          return result;
        }
        --stackPtr;
        if (__int_as_float(stackPtr->y) > maxQueryDistSquare) continue;
        nodeID = stackPtr->x;
        break;
      }
    }
  }



//   // inline __device__
//   // float3 project(box3f box, float3 point)
//   // { return max(min(point,box.upper),box.lower); }
  
//   // inline __device__
//   // float sqrDistance(box3f box, float3 point)
//   // { return sqrDistance(project(box,point),point); }

//   // inline __device__
//   // float sqrDistance(BinaryBVH<float,3>::Node node, float3 point)
//   // { return sqrDistance(node.bounds,point); }

//   /*! 'fcp' = "find closest point", on a binary BVH. Given an input
//     query point, and a BVH over point data, find the (index of) the
//     data point that is closest to the given query point. Function
//     also takes a maximum query distance; all data points with
//     distance > this minimum distance will get ignored. If no fcp can
//     be found in given query radius, -1 will be returned 
    
//     Careful, the max query distance is specified as the SQUARE of the
//     maximum search distance because that'll allow to avoid various
//     expensive sqrt operations
//   */
//   inline __device__
//   int fcp(const BinaryBVH<float,3> bvh,
//           const float3   *dataPoints,
//           const float3    query,
//           /* in: SQUARE of max search distance; out: sqrDist of closest point */
//           float          &maxQueryDistSquare)
//   {
// #if 1
//     int result = -1;
    
//     int2 stackBase[32], *stackPtr = stackBase;
//     int nodeID = 0;
//     int offset = 0;
//     int count  = 0;
//     while (true) {
//       while (true) {
//         offset = bvh.nodes[nodeID].offset;
//         count  = bvh.nodes[nodeID].count;
//         if (count>0)
//           // leaf
//           break;
//         BinaryBVH<float,3>::Node child0 = bvh.nodes[offset+0];
//         BinaryBVH<float,3>::Node child1 = bvh.nodes[offset+1];
//         float dist0 = sqrDistance(child0,query);
//         float dist1 = sqrDistance(child1,query);
//         int closeChild = offset + ((dist0 > dist1) ? 1 : 0);
//         if (dist1 <= maxQueryDistSquare) {
//           float dist = max(dist0,dist1);
//           int distBits = __float_as_int(dist);
//           *stackPtr++ = make_int2(closeChild^1,distBits);
//         }
//         if (min(dist0,dist1) > maxQueryDistSquare) {
//           count = 0;
//           break;
//         }
//         nodeID = closeChild;
//       }
//       for (int i=0;i<count;i++) {
//         int primID = bvh.primIDs[offset+i];
//         float dist2 = sqrDistance(dataPoints[primID],query);
//         if (dist2 >= maxQueryDistSquare) continue;
//         maxQueryDistSquare = dist2;
//         result             = primID;
//       }
//       while (true) {
//         if (stackPtr == stackBase)
//           return result;
//         --stackPtr;
//         if (__int_as_float(stackPtr->y) > maxQueryDistSquare) continue;
//         nodeID = stackPtr->x;
//         break;
//       }
//     }
// #endif
//   }

  template<int N>
  struct ChildOrder {
    inline __device__ void clear(int i) { v[i] = (uint64_t)-1; }
    inline __device__ void set(int i, float dist, uint32_t payload)
    { v[i] = (uint64_t(__float_as_int(dist))<<32) | payload; }
    
    uint64_t v[N];
  };
  
  template<int N>
  inline __device__ void sort(ChildOrder<N> &children)
  {
#pragma unroll
    for (int i=N-1;i>0;--i) {
#pragma unroll
      for (int j=0;j<i;j++) {
        uint64_t c0 = children.v[j+0];
        uint64_t c1 = children.v[j+1];
        children.v[j+0] = min(c0,c1);
        children.v[j+1] = max(c0,c1);
      }
    }
  }
  
//   /*! 'fcp' = "find closest point", on a binary BVH. Given an input
//     query point, and a BVH over point data, find the (index of) the
//     data point that is closest to the given query point. Function
//     also takes a maximum query distance; all data points with
//     distance > this minimum distance will get ignored. If no fcp can
//     be found in given query radius, -1 will be returned */
//   template<int N>
//   inline __device__
//   int fcp(/*! the bvh that is built over hte points */
//           const WideBVH<float,3,N> bvh,
          
//           /*! the data points that the BVH is actually built over */
//           const float3    *dataPoints,
          
//           /*! the query position, for which we try to find the
//             closest point */
//           const float3     query,
          
//           /*! in: SQUARE of max search distance; out: sqrDist of closest point */
//           float           &maxQueryDistSquare)
//   {
// #if 1
//     int result = -1;

//     enum { stackSize = 64 };
//     uint64_t stackBase[stackSize], *stackPtr = stackBase;
//     int nodeID = 0;
//     ChildOrder<N> childOrder;
//     while (true) {
//       while (true) {
//         while (nodeID == -1) {
//           if (stackPtr == stackBase)
//             return result;
//           uint64_t tos = *--stackPtr;
//           if (__int_as_float(tos>>32) > maxQueryDistSquare)
//             continue;
//           nodeID = (uint32_t)tos;
//           // pop....
//         }
//         if (nodeID & (1<<31))
//           break;
        
//         const typename WideBVH<float,3,N>::Node &node = bvh.nodes[nodeID];
// #pragma unroll(N)
//         for (int c=0;c<N;c++) {
//           const auto child = node.children[c];
//           if (!node.children[c].valid)
//             childOrder.clear(c);
//           else {
//             float dist2 = sqrDistance(child.bounds,query);
//             if (dist2 > maxQueryDistSquare) 
//               childOrder.clear(c);
//             else {
//               uint32_t payload
//                 = child.count
//                 ? ((1<<31)|(nodeID<<log_of<N>::value)|c)
//                 : child.offset;
//               childOrder.set(c,dist2,payload);
//             }
//           }
//         }
//         sort(childOrder);
// #pragma unroll
//         for (int c=N-1;c>0;--c) {
//           uint64_t coc = childOrder.v[c];
//           if (coc != uint64_t(-1)) {
//             *stackPtr++ = coc;
//             // if (stackPtr - stackBase == stackSize)
//             //   printf("stack overrun!\n");
//           }
//         }
//         if (childOrder.v[0] == uint64_t(-1)) {
//           nodeID = -1;
//           continue;
//         }
//         nodeID = uint32_t(childOrder.v[0]);
//       }
      
//       int c = nodeID & ((1<<log_of<N>::value)-1);
//       int n = (nodeID & 0x7fffffff)  >> log_of<N>::value;
//       int offset = bvh.nodes[n].children[c].offset;
//       int count  = bvh.nodes[n].children[c].count;
//       for (int i=0;i<count;i++) {
//         int primID = bvh.primIDs[offset+i];
//         float dist2 = sqrDistance(dataPoints[primID],query);
//         if (dist2 >= maxQueryDistSquare) continue;
//         maxQueryDistSquare = dist2;
//         result             = primID;
//       }
//       nodeID = -1;
//     }
// #endif
//   }


//   /*! for convenience, a fcp variant that doesn't have a max query
//       dist, and returns only the int */
//   template<typename bvh_t>
//   inline __device__
//   int fcp(/*! the bvh that is built over hte points */
//           bvh_t bvh,
          
//           /*! the data points that the BVH is actually built over */
//           const float3 *dataPoints,
          
//           /*! the query position, for which we try to find the
//             closest point */
//           const float3 query)
//   {
//     float sqrMaxQueryDist = INFINITY;
//     return fcp(bvh,dataPoints,query);
//   }
} // ::cuBQL

