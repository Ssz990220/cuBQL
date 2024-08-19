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

#include "cuBQL/queries/fcp.h"

namespace cuBQL {

  /*! HEAP-based implementation of k-nearest neighbors. This
      implementation guarantted the following:

      - "count" is the number of found items within this struct's
        items[] list. It does *NOT* guarantee that thees items will be
        stored in items[0...count]; they may be stored in nay slots

      - maxDist2 is the (current) range of the qhery ball: if K items
        have been found, this iwll be the distnace to the furtherst of
        these K items; otherwise it's the initial max query distance

      - if count < K (ie, not all K could be found), then some of the
        items[] slots may be unused; and those that *are* unsued will
        have an item ID < 0.

      it does explicitly NOT guarattee that items in the items[] array
      would be sorted by distance

      it does explicitly NOT guarantee that slots in the items[] array
      would be used from the front; ie, if count<K items[0] will
      typically be a INVALID item.
  */
  template<int K>
  struct KNNResults {

    inline __device__ void  clear(float initialMaxDist);
    inline __device__ float insert(float dist, int ID);
    inline __device__ float getDist(int i) const;
    inline __device__ uint32_t getItem(int i) const;
    float    maxDist2;
  private:
    inline __device__ static uint64_t makeItem(float dist, int itemID);
  public:
    inline __device__ void printCurrent()
    {
      printf("kNearest, count = %i, maxDist2 = %f\n",count,maxDist2);
      for (int i=0;i<K;i++) {
        if (i < count)
          printf("-%5i",i);
        else
          printf("-[%3i]",i);
        printf("\tdist = %f\tID = %i\n",
               getDist(i),getItem(i));
      }
    }
  private:
    int      count;
    uint64_t items[K];
  };

  template<int K> __device__
  void KNNResults<K>::clear(float initialMaxDist)
  {
    count = 0;
#pragma runroll
    for (int i=0;i<K;i++)
      items[i] = makeItem(INFINITY,-2);//uint64_t(-1);
    maxDist2 = initialMaxDist;
  }
  
  template<int K> __device__
  float KNNResults<K>::insert(float dist, int ID)
  {
    if (dist > maxDist2) 
      return maxDist2;
    
    uint64_t item = makeItem(dist,ID);
    int pos = 0;
    while (1) {
      // pos of first child in heap
      int cc = 2*pos+1;
      if (cc >= K)
        // does not have any children
        break;
      uint64_t cItem = items[cc];

      int c1 = cc+1;
      if (c1 < K && items[c1] > cItem) {
        cc = c1;
        cItem = items[c1];
      }
      
      if (cItem <= item)
        break;
      items[pos] = cItem;
      pos = cc;
    }
    items[pos] = item;
    count = min(K,count+1);
    maxDist2 = min(maxDist2,getDist(0));
    return maxDist2;
  }
  
  template<int K> __device__
  float KNNResults<K>::getDist(int i) const
  {
    return __int_as_float(items[i] >> 32);
  }
  
  template<int K> __device__
  uint32_t KNNResults<K>::getItem(int i) const
  {
    return uint32_t(items[i]);
  }
  
  template<int K> __device__
  uint64_t KNNResults<K>::makeItem(float dist, int itemID) 
  {
    // compiler will turn that into insertfield op
    return uint32_t(itemID) | (uint64_t(__float_as_uint(dist)) << 32);
  }
  






  template<typename ResultList, int D>
  inline __device__
  void knn(ResultList &results,
           const BinaryBVH<float,D> bvh,
#if USE_BOXES
           const box_t<float,D>    *prims,
#else
           const vec_t<float,D>    *prims,
#endif
           const vec_t<float,D>     query
           // ,
          /* in: SQUARE of max search distance; out: sqrDist of closest point */
          // float          &maxQueryDistSquare
#if DO_STATS
          , Stats *d_stats
#endif
          )
  {
    
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
        if (dist1 < results.maxDist2) {
          float dist = max(dist0,dist1);
          int distBits = __float_as_int(dist);
          *stackPtr++ = make_int2(closeChild^1,distBits);
        }
        if (min(dist0,dist1) > results.maxDist2) {
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
        if (dist2 >= results.maxDist2) continue;
        results.insert(dist2,primID);
      }
      while (true) {
        if (stackPtr == stackBase) {
#if DO_STATS
          atomicAdd(&d_stats->numNodes,numNodes);
          atomicAdd(&d_stats->numPrims,numPrims);
#endif
          // int tid = threadIdx.x+blockIdx.x*blockDim.x;
          // bool dbg = tid == 1117;
          // if (dbg) {
          //   printf("numnodes %i numprims %i\n",numNodes,numPrims);
          //   printf("query %f %f result radius2 %f \n",query[0],query[1],results.maxDist2);
      
          //   results.printCurrent();
          // }
          

          return;
        }
        --stackPtr;
        if (__int_as_float(stackPtr->y) > results.maxDist2) continue;
        nodeID = stackPtr->x;
        break;
      }
    }
  }

  
//   template<
//   inline __device__
//   int knn(const BinaryBVH<float,3> bvh,
          
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

//   template<int N>
//   struct ChildOrder {
//     inline __device__ void clear(int i) { v[i] = (uint64_t)-1; }
//     inline __device__ void set(int i, float dist, uint32_t payload)
//     { v[i] = (uint64_t(__float_as_int(dist))<<32) | payload; }
    
//     uint64_t v[N];
//   };
  
//   template<int N>
//   inline __device__ void sort(ChildOrder<N> &children)
//   {
// #pragma unroll
//     for (int i=N-1;i>0;--i) {
// #pragma unroll
//       for (int j=0;j<i;j++) {
//         uint64_t c0 = children.v[j+0];
//         uint64_t c1 = children.v[j+1];
//         children.v[j+0] = min(c0,c1);
//         children.v[j+1] = max(c0,c1);
//       }
//     }
//   }
  
//   /*! 'knn' = "find closest point", on a binary BVH. Given an input
//     query point, and a BVH over point data, find the (index of) the
//     data point that is closest to the given query point. Function
//     also takes a maximum query distance; all data points with
//     distance > this minimum distance will get ignored. If no knn can
//     be found in given query radius, -1 will be returned */
//   template<int N>
//   inline __device__
//   int knn(/*! the bvh that is built over hte points */
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


//   /*! for convenience, a knn variant that doesn't have a max query
//       dist, and returns only the int */
//   template<typename bvh_t>
//   inline __device__
//   int knn(/*! the bvh that is built over hte points */
//           bvh_t bvh,
          
//           /*! the data points that the BVH is actually built over */
//           const float3 *dataPoints,
          
//           /*! the query position, for which we try to find the
//             closest point */
//           const float3 query)
//   {
//     float sqrMaxQueryDist = INFINITY;
//     return knn(bvh,dataPoints,query);
//   }
} // ::cuBQL

