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
#include <cub/cub.cuh>

namespace cuBQL {
  namespace computeSAH_impl {
    template<typename T, int D>
    __global__
    void computeNodeCosts(BinaryBVH<T,D> bvh, float *nodeCosts)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= bvh.numNodes) return;

      if (nodeID == 1) { nodeCosts[nodeID] = 0.f; return; }

      auto node = bvh.nodes[nodeID];
      float area = surfaceArea(node.bounds) / surfaceArea(bvh.nodes[0].bounds);
      if (node.admin.count == 0)
        nodeCosts[nodeID] = area;
      else 
        nodeCosts[nodeID] = area * node.admin.count;
    }
    
    // template<typename T, int D>
    // float computeSAH(const BinaryBVH<T,D> &bvh)
    // { throw std::runtime_error("cannot compute sah for num dims != 3"); }
    
    template<typename T>
    float computeSAH(const BinaryBVH<T,3> &bvh)
    {
      float *nodeCosts;
      float *reducedCosts;
      CUBQL_CUDA_CALL(MallocManaged((void**)&nodeCosts,bvh.numNodes*sizeof(float)));
      CUBQL_CUDA_CALL(MallocManaged((void**)&reducedCosts,sizeof(float)));
      computeNodeCosts<<<divRoundUp(int(bvh.numNodes),1024),1024>>>(bvh,nodeCosts);

      CUBQL_CUDA_SYNC_CHECK();
      
      // Determine temporary device storage requirements
      void     *d_temp_storage = NULL;
      size_t   temp_storage_bytes = 0;
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                             nodeCosts, reducedCosts, bvh.numNodes);
      // Allocate temporary storage
      CUBQL_CUDA_CALL(Malloc(&d_temp_storage, temp_storage_bytes));
      // Run sum-reduction
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                             nodeCosts, reducedCosts, bvh.numNodes);
      
      CUBQL_CUDA_SYNC_CHECK();
      float result = reducedCosts[0];

      CUBQL_CUDA_CALL(Free(d_temp_storage));
      CUBQL_CUDA_CALL(Free(nodeCosts));
      CUBQL_CUDA_CALL(Free(reducedCosts));
      return result;
    }

    template<typename T, int D, int N>
    __global__
    void computeNodeCosts(WideBVH<T,D,N> bvh, float *nodeCosts)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= bvh.numNodes) return;

      auto &node = bvh.nodes[nodeID];
      float area = 0.f;
      for (int i=0;i<N;i++) {
        box_t<T,D> box = node.children[i].bounds;
        if (box.lower.x > box.upper.x) continue;
        area += surfaceArea(box);
      }
      box_t<T,D> rootBox; rootBox.set_empty();
      for (int i=0;i<N;i++)
        rootBox.grow(bvh.nodes[0].children[i].bounds);
      area /= surfaceArea(rootBox);
      nodeCosts[nodeID] = area;
    }

    template<typename T, int D, int N>
    float computeSAH(const WideBVH<T,D,N> &bvh)
    {
      float *nodeCosts;
      float *reducedCosts;
      CUBQL_CUDA_CALL(MallocManaged((void**)&nodeCosts,bvh.numNodes*sizeof(float)));
      CUBQL_CUDA_CALL(MallocManaged((void**)&reducedCosts,sizeof(float)));
      computeNodeCosts<<<divRoundUp(int(bvh.numNodes),1024),1024>>>(bvh,nodeCosts);

      CUBQL_CUDA_SYNC_CHECK();
      
      // Determine temporary device storage requirements
      void     *d_temp_storage = NULL;
      size_t   temp_storage_bytes = 0;
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                             nodeCosts, reducedCosts, bvh.numNodes);
      // Allocate temporary storage
      CUBQL_CUDA_CALL(Malloc(&d_temp_storage, temp_storage_bytes));
      // Run sum-reduction
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                             nodeCosts, reducedCosts, bvh.numNodes);
      
      CUBQL_CUDA_SYNC_CHECK();
      float result = reducedCosts[0];

      CUBQL_CUDA_CALL(Free(d_temp_storage));
      CUBQL_CUDA_CALL(Free(nodeCosts));
      CUBQL_CUDA_CALL(Free(reducedCosts));
      return result;
    }
    
  }
  
  // ------------------------------------------------------------------
  
  /*! computes the SAH cost of a already built BinaryBVH. This is
      often a useful metric for how "good" a BVH is */
  template<typename T, int D>
  float computeSAH(const BinaryBVH<T,D> &bvh)
  { throw std::runtime_error("cannot compute SAH for num dims != 3");}
  
  template<typename T>
  float computeSAH(const BinaryBVH<T,3> &bvh)
  { return computeSAH_impl::computeSAH(bvh); }
  // template<typename T, int D>
  // float computeSAH(const BinaryBVH<T,D> &bvh);
  
  template<typename T, int D, int N>
  float computeSAH(const WideBVH<T,D,N> &bvh)
  { throw std::runtime_error("cannot compute SAH for num dims != 3");}
  
  /*! computes the SAH cost of a already built WideBVH. This is often a
      useful metric for how "good" a BVH is */
  template<typename T, int N>
  float computeSAH(const WideBVH<T,3,N> &bvh)
  { return computeSAH_impl::computeSAH(bvh); }
  // template<typename T, int D, int N>
  // float computeSAH(const WideBVH<T,D,N> &bvh);

  // template<typename T, int D>
  // float computeSAH(const BinaryBVH<T,D> &bvh)
  // {
  //   if (D == 3)
  //     // SAH only defined for 3-dimensional data...
  //     return computeSAH_impl::computeSAH(bvh);
  //   else
  //     throw std::runtime_error("cannot compute SAH for this type of BVH");
  // }
  
} // :: cuBQL
