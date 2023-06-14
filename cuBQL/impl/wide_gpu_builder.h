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

// #include <vector>

#if CUBQL_GPU_BUILDER_IMPLEMENTATION

namespace cuBQL {
  namespace gpuBuilder_impl {
    
    struct CollapseInfo {
      // careful: 'isWideRoot' and ''binaryRoot' get written to in
      // parallel by differnet threads; they must be in different atomic
      // words.
      struct {
        int32_t  parent:31;
        uint32_t isWideRoot:1;
      };
      /*! for *wide* nodes: the ID of the binary node that is the root
          of the treelet that this node maps to */
      int32_t binaryRoot;
      
      /*! for *binary* nodes that re treelet root nodes: the ID of the
        wide node that it maps to */
      int     wideNodeID;
    };
  
    template<typename box_t>
    __global__
    void collapseInit(int *d_numWideNodes,
                      CollapseInfo *d_infos,
                      BinaryBVH<box_t> bvh)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= bvh.numNodes) return;

      if (tid == 0) {
        *d_numWideNodes  =  1;
        d_infos[0].parent = -1;
        d_infos[0].isWideRoot = 1;
        d_infos[0].wideNodeID = 0;
        d_infos[0].binaryRoot = -1;
      }

      auto &node = bvh.nodes[tid];
      if (node.count > 0)
        // leaf node
        return;
    
      // _could_ write this as a int4 ... we know it'll have to be
      // 128-bit aligned
      d_infos[node.offset+0].isWideRoot = 0;
      d_infos[node.offset+0].parent = tid;
      d_infos[node.offset+0].binaryRoot = -1;
      d_infos[node.offset+1].isWideRoot = 0;
      d_infos[node.offset+1].parent = tid;
      d_infos[node.offset+1].binaryRoot = -1;
    }

    template<typename box_t, int N>
    __global__
    void collapseSummarize(int *d_numWideNodes,
                           CollapseInfo *d_infos,
                           BinaryBVH<box_t> bvh)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= bvh.numNodes) return;
      if (tid == 1)
        // bvh.node[1] is always unused
        return;
    
      int depth  = 0;
      {
        int nodeID = tid;
        while (nodeID > 0) {
          depth++;
          nodeID = d_infos[nodeID].parent;
        }
      }

      const bool isWideNodeRoot
        =  /* inner node: */
        (bvh.nodes[tid].count == 0)
        && /* on right level*/
        ((depth % (log_of<N>::value)) == 0)

        || /* special case: single-node BVH */
        (bvh.numNodes == 1);

      if (!isWideNodeRoot) 
        return;

      const int wideNodeID
        = (tid == 0)
        ? 0
        : atomicAdd(d_numWideNodes,1);
      d_infos[wideNodeID].binaryRoot = tid;
      d_infos[tid].isWideRoot = true;
      d_infos[tid].wideNodeID = wideNodeID;
    }


    template<typename box_t, int N>
    __global__
    void collapseExecute(CollapseInfo *d_infos,
                         WideBVH<box_t,N> wideBVH,
                         BinaryBVH<box_t>  binary)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= wideBVH.numNodes)
        return;

      int nodeStack[5], *stackPtr = nodeStack;
      int binaryRoot = d_infos[tid].binaryRoot;
      *stackPtr++ = binaryRoot;

      typename WideBVH<box_t,N>::Node &target = wideBVH.nodes[tid];
      int numWritten = 0;
      while (stackPtr > nodeStack) {
        int nodeID = *--stackPtr;
        auto &node = binary.nodes[nodeID];
        if ((node.count > 0) ||
            ((nodeID != binaryRoot) && d_infos[nodeID].isWideRoot)) {
          target.children[numWritten].bounds = node.bounds;
          if (node.count) {
            target.children[numWritten].offset = node.offset;
          } else {
            target.children[numWritten].offset = d_infos[nodeID].wideNodeID;
          }
          target.children[numWritten].count  = node.count;
          target.children[numWritten].valid  = 1;
          numWritten++;
        } else {
          *stackPtr++ = node.offset+0;
          *stackPtr++ = node.offset+1;
        }
      }
      while (numWritten < N) {
        target.children[numWritten].bounds.set_empty();
        // lower
        //   = make_float3(+INFINITY,+INFINITY,+INFINITY);
        // target.children[numWritten].bounds.upper
        //   = make_float3(-INFINITY,-INFINITY,-INFINITY);
        target.children[numWritten].offset = (uint32_t)-1;
        target.children[numWritten].count  = (uint32_t)-1;
        target.children[numWritten].valid  = 0;
        ++numWritten;
      }
    }

    template<typename box_t, int N>
    void gpuBuilder(WideBVH<box_t,N>  &wideBVH,
                    const box_t *boxes,
                    uint32_t     numBoxes,
                    BuildConfig  buildConfig,
                    cudaStream_t s)
    {
      
      BinaryBVH<box_t> binaryBVH;
      gpuBuilder(binaryBVH,boxes,numBoxes,buildConfig,s);

      int          *d_numWideNodes;
      CollapseInfo *d_infos;
      _ALLOC(d_numWideNodes,1,s);
      _ALLOC(d_infos,binaryBVH.numNodes,s);
    
      collapseInit<<<divRoundUp((int)binaryBVH.numNodes,1024),1024,0,s>>>
        (d_numWideNodes,d_infos,binaryBVH);

      collapseSummarize<box_t,N><<<divRoundUp((int)binaryBVH.numNodes,1024),1024,0,s>>>
        (d_numWideNodes,d_infos,binaryBVH);
      CUBQL_CUDA_CALL(StreamSynchronize(s));

      CUBQL_CUDA_CALL(MemcpyAsync(&wideBVH.numNodes,d_numWideNodes,
                                  sizeof(int),cudaMemcpyDefault,s));
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _ALLOC(wideBVH.nodes,wideBVH.numNodes,s);

      collapseExecute<<<divRoundUp((int)wideBVH.numNodes,1024),1024,0,s>>>
        (d_infos,wideBVH,binaryBVH);

      wideBVH.numPrims = binaryBVH.numPrims;
      wideBVH.primIDs  = binaryBVH.primIDs;
      binaryBVH.primIDs = 0;
    
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _FREE(d_infos,s);
      _FREE(d_numWideNodes,s);
      free(binaryBVH,s);
    }

    template<typename box_t, int N>
    __global__
    void computeNodeCosts(WideBVH<box_t,N> bvh, float *nodeCosts)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= bvh.numNodes) return;

      auto &node = bvh.nodes[nodeID];
      float area = 0.f;
      for (int i=0;i<N;i++) {
        box_t box = node.children[i].bounds;
        if (box.lower.x > box.upper.x) continue;
        area += surfaceArea(box);
      }
      box_t rootBox; rootBox.set_empty();
      for (int i=0;i<N;i++)
        rootBox.grow(bvh.nodes[0].children[i].bounds);
      area /= surfaceArea(rootBox);
      nodeCosts[nodeID] = area;
    }

    template<typename box_t, int N>
    float computeSAH(const WideBVH<box_t,N> &bvh)
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
    
  } // ::cuBQL::gpuBuilder_impl

  template<typename box_t, int N>
  void gpuBuilder(WideBVH<box_t,N>   &bvh,
                  const box_t *boxes,
                  uint32_t     numBoxes,
                  BuildConfig  buildConfig,
                  cudaStream_t s)
  {
    gpuBuilder_impl::gpuBuilder(bvh,boxes,numBoxes,buildConfig,s);
  }

  template<typename box_t, int N>
  void free(WideBVH<box_t,N>   &bvh,
            cudaStream_t s)
  {
    CUBQL_CUDA_CALL(StreamSynchronize(s));
    CUBQL_CUDA_CALL(FreeAsync(bvh.primIDs,s));
    CUBQL_CUDA_CALL(FreeAsync(bvh.nodes,s));
    CUBQL_CUDA_CALL(StreamSynchronize(s));
    bvh.primIDs = 0;
  }

  template<typename box_t, int N>
  float computeSAH(const WideBVH<box_t,N> &bvh)
  {
    return gpuBuilder_impl::computeSAH(bvh);
  }
  
} // :: cuBQL

#endif
