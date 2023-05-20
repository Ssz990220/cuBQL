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

#include "cuBQL/impl/builder_common.h"

namespace cuBQL {
  namespace gpuBuilder_impl {

    struct PrimState {
      union {
        /* careful with this order - this is intentionally chosen such
           that all item with nodeID==-1 will end up at the end of the
           list; and all others will be sorted by nodeID */
        struct {
          uint64_t primID:31; //!< prim we're talking about
          uint64_t done  : 1;
          uint64_t nodeID:32; //!< node the given prim is (currently) in.
        };
        uint64_t bits;
      };
    };

    struct CUBQL_ALIGN(16) TempNode {
      union {
        struct {
          AtomicBox<box3f> centBounds;
          uint32_t         count;
          uint32_t         unused;
        } openBranch;
        struct {
          uint32_t offset;
          int      dim;
          uint32_t tieBreaker;
          float    pos;
        } openNode;
        struct {
          uint32_t offset;
          uint32_t count;
          uint32_t unused[2];
        } doneNode;
      };
    };
    
    __global__
    void initState(BuildState *buildState,
                   NodeState  *nodeStates,
                   TempNode   *nodes)
    {
      // buildState->nodes = nodes;
      buildState->numNodes = 2;
      
      nodeStates[0]             = OPEN_BRANCH;
      nodes[0].openBranch.count = 0;
      nodes[0].openBranch.centBounds.set_empty();

      nodeStates[1]            = DONE_NODE;
      nodes[1].doneNode.offset = 0;
      nodes[1].doneNode.count  = 0;
    }
  
    __global__ void initPrims(TempNode    *nodes,
                              PrimState   *primState,
                              const box3f *primBoxes,
                              uint32_t     numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;
      
      auto &me = primState[primID];
      me.primID = primID;
                                                    
      const box3f box = primBoxes[primID];
      if (box.get_lower(0) <= box.get_upper(0)) {
        me.nodeID = 0;
        me.done   = false;
        // this could be made faster by block-reducing ...
        atomicAdd(&nodes[0].openBranch.count,1);
        atomic_grow(nodes[0].openBranch.centBounds,box);
      } else {
        me.nodeID = (uint32_t)-1;
        me.done   = true;
      }
      // auto asDone = me;
      // asDone.done = true;
      // printf(" prim %i me %lx asDone %lx\n",primID,me.bits,asDone.bits);
    }

    __global__
    void selectSplits(BuildState *buildState,
                      NodeState  *nodeStates,
                      TempNode   *nodes,
                      uint32_t    numNodes,
                      BuildConfig buildConfig)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      NodeState &nodeState = nodeStates[nodeID];
      if (nodeState == DONE_NODE)
        // this node was already closed before
        return;
      
      if (nodeState == OPEN_NODE) {
        // this node was open in the last pass, can close it.
        nodeState   = DONE_NODE;
        int offset  = nodes[nodeID].openNode.offset;
        auto &done  = nodes[nodeID].doneNode;
        done.count  = 0;
        done.offset = offset;
        return;
      }
      
      auto in = nodes[nodeID].openBranch;
      if (in.count <= buildConfig.makeLeafThreshold) {
        auto &done  = nodes[nodeID].doneNode;
        done.count  = in.count;
        // set this to max-value, so the prims can later do atomicMin
        // with their position ion the leaf list; this value is
        // greater than any prim position.
        done.offset = (uint32_t)-1;
        nodeState   = DONE_NODE;
      } else {
        float widestWidth = 0.f;
        int   widestDim   = -1;
#pragma unroll
        for (int d=0;d<3;d++) {
          float width = in.centBounds.get_upper(d) - in.centBounds.get_lower(d);
          if (width <= widestWidth)
            continue;
          widestWidth = width;
          widestDim   = d;
        }
      
        auto &open = nodes[nodeID].openNode;
        open.dim = widestDim;
        if (widestDim >= 0) 
          open.pos = in.centBounds.get_center(widestDim);
        // this will be epensive - could make this faster by block-reducing
        open.offset = atomicAdd(&buildState->numNodes,2);
#pragma unroll
        for (int side=0;side<2;side++) {
          const int childID = open.offset+side;
          auto &child = nodes[childID].openBranch;
          child.centBounds.set_empty();
          child.count         = 0;
          nodeStates[childID] = OPEN_BRANCH;
        }
        nodeState = OPEN_NODE;
      }
    }

    __global__
    void updatePrims(NodeState       *nodeStates,
                     TempNode        *nodes,
                     PrimState       *primStates,
                     const box3f     *primBoxes,
                     int numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;

      auto &me = primStates[primID];
      if (me.done) return;
      
      auto ns = nodeStates[me.nodeID];
      if (ns == DONE_NODE) {
        // node became a leaf, we're done.
        me.done = true;
        return;
      }

      auto &split = nodes[me.nodeID].openNode;
      const box3f primBox = primBoxes[me.primID];
      int side = 0;
      if (split.dim == -1) {
        // could block-reduce this, but will likely not happen often, anyway
        side = (atomicAdd(&split.tieBreaker,1) & 1);
      } else {
        const float center = 0.5f*(primBox.get_lower(split.dim)+
                                   primBox.get_upper(split.dim));
        side = (center >= split.pos);
      }
      int newNodeID = split.offset+side;
      auto &myBranch = nodes[newNodeID].openBranch;
      atomicAdd(&myBranch.count,1);
      atomic_grow(myBranch.centBounds,primBox);
      me.nodeID = newNodeID;
    }

    /* given a sorted list of {nodeID,primID} pairs, this kernel does
       two things: a) it extracts the 'primID's and puts them into the
       bvh's primIDs[] array; and b) it writes, for each leaf nod ein
       the nodes[] array, the node.offset value to point to the first
       of this nodes' items in that bvh.primIDs[] list. */
    __global__
    void writePrimsAndLeafOffsets(TempNode        *nodes,
                                  uint32_t        *bvhItemList,
                                  PrimState       *primStates,
                                  int              numPrims)
    {
      const int offset = threadIdx.x+blockIdx.x*blockDim.x;
      if (offset >= numPrims) return;

      auto &ps = primStates[offset];
      bvhItemList[offset] = ps.primID;
      
      if ((int)ps.nodeID < 0)
        /* invalid prim, just skip here */
        return;
      auto &node = nodes[ps.nodeID];
      atomicMin(&node.doneNode.offset,offset);
    }


    /* writes main phase's temp nodes into final bvh.nodes[]
       layout. actual bounds of that will NOT yet bewritten */
    __global__
    void writeNodes(BinaryBVH::Node *finalNodes,
                    TempNode  *tempNodes,
                    int        numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      finalNodes[nodeID].offset = tempNodes[nodeID].doneNode.offset;
      finalNodes[nodeID].count  = tempNodes[nodeID].doneNode.count;
    }

    /*! pretty-prints the current bvh for debugging; must use managed
      memory or download bvh data to host */
    void printBVH(TempNode *nodes, int numNodes,
                  uint32_t *primIDs, int numPrims,
                  int nodeID=0, int indent=0)
    {
      auto &node = nodes[nodeID].doneNode;
      for (int i=0;i<indent;i++) std::cout << " ";
      std::cout << "Node, offset = " << node.offset
                << ", count = " << node.count << std::endl;
      if (node.count) {
        for (int i=0;i<indent;i++) std::cout << " ";
        std::cout << "Leaf:";
        for (int i=0;i<node.count;i++)
          std::cout << " " << primIDs[node.offset+i];
        std::cout << std::endl;
      } else {
        printBVH(nodes,numNodes,primIDs,numPrims,
                 node.offset+0,indent+1);
        printBVH(nodes,numNodes,primIDs,numPrims,
                 node.offset+1,indent+1);
      }
    }

    void build(BinaryBVH &bvh,
               const box3f *boxes,
               int numPrims,
               BuildConfig  buildConfig,
               cudaStream_t s)
    {
      // ==================================================================
      // do build on temp nodes
      // ==================================================================
      TempNode   *tempNodes = 0;
      NodeState  *nodeStates = 0;
      PrimState  *primStates = 0;
      BuildState *buildState = 0;
      _ALLOC(tempNodes,2*numPrims,s);
      _ALLOC(nodeStates,2*numPrims,s);
      _ALLOC(primStates,numPrims,s);
      _ALLOC(buildState,1,s);
      initState<<<1,1,0,s>>>(buildState,
                             nodeStates,
                             tempNodes);
      initPrims<<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,
         primStates,boxes,numPrims);
      // (tempNodes.data(),
      //  primStates.data(),boxes,numPrims);

      int numDone = 0;
      int numNodes;

      // ------------------------------------------------------------------      
      while (true) {
        CUBQL_CUDA_CALL(MemcpyAsync(&numNodes,&buildState->numNodes,
                                    sizeof(numNodes),cudaMemcpyDeviceToHost,s));
        CUBQL_CUDA_CALL(StreamSynchronize(s));
        // CUBQL_CUDA_CALL(Memcpy(&numNodes,&buildState.data()->numNodes,
        //                        sizeof(numNodes),cudaMemcpyDeviceToHost));
        if (numNodes == numDone)
          break;

        selectSplits<<<divRoundUp(numNodes,1024),1024,0,s>>>
          (buildState,
           nodeStates,tempNodes,numNodes,
           buildConfig);
        // (buildState.data(),
        //  nodeStates.data(),tempNodes.data(),numNodes,
        //  maxLeafSize);
        
        numDone = numNodes;

        updatePrims<<<divRoundUp(numPrims,1024),1024,0,s>>>
          (nodeStates,tempNodes,
           primStates,boxes,numPrims);
        // (nodeStates.data(),tempNodes.data(),
        //  primStates.data(),boxes,numPrims);
      }
      // ==================================================================
      // sort {item,nodeID} list
      // ==================================================================
      
      // set up sorting of prims
      uint8_t *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      // CUDAArray<PrimState> sortedPrimStates(numPrims);
      PrimState *sortedPrimStates;
      _ALLOC(sortedPrimStates,numPrims,s);
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,//.data(),
                                     (uint64_t*)sortedPrimStates,//.data(),
                                     numPrims,32,64,s);
      // CUBQL_CUDA_CALL(Malloc(&d_temp_storage,temp_storage_bytes));
      _ALLOC(d_temp_storage,temp_storage_bytes,s);
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,//.data(),
                                     (uint64_t*)sortedPrimStates,//.data(),
                                     numPrims,32,64,s);
      // CUBQL_CUDA_CALL(Free(d_temp_storage));
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _FREE(d_temp_storage,s);
      // primStates.free();
      // ==================================================================
      // allocate and write BVH item list, and write offsets of leaf nodes
      // ==================================================================

      bvh.numPrims = numPrims;
      // CUBQL_CUDA_CALL(Malloc(&bvh.primIDs,numPrims*sizeof(int)));
      _ALLOC(bvh.primIDs,numPrims,s);
      writePrimsAndLeafOffsets<<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,bvh.primIDs,sortedPrimStates,numPrims);
      // (tempNodes.data(),bvh.primIDs,sortedPrimStates.data(),numPrims);

      // ==================================================================
      // allocate and write final nodes
      // ==================================================================
      bvh.numNodes = numNodes;
      // CUBQL_CUDA_CALL(Malloc(&bvh.nodes,numNodes*sizeof(BinaryBVH::Node)));
      _ALLOC(bvh.nodes,numNodes,s);
      writeNodes<<<divRoundUp(numNodes,1024),1024,0,s>>>
        // (bvh.nodes,tempNodes.data(),numNodes);
        (bvh.nodes,tempNodes,numNodes);
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _FREE(sortedPrimStates,s);
      _FREE(tempNodes,s);
      _FREE(nodeStates,s);
      _FREE(primStates,s);
      _FREE(buildState,s);
    }

    __global__ void
    refit_init(const BinaryBVH::Node *nodes,
               uint32_t              *refitData,
               int numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;
      if (nodeID == 0)
        refitData[0] = 0;
      const auto &node = nodes[nodeID];
      if (node.count) return;
      refitData[node.offset+0] = nodeID << 1;
      refitData[node.offset+1] = nodeID << 1;
    }
    
    __global__
    void refit_run(BinaryBVH bvh,
                   uint32_t *refitData,
                   const box3f *boxes)
    {
      int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= bvh.numNodes) return;
      if (nodeID == 1) return;
      
      BinaryBVH::Node *node = &bvh.nodes[nodeID];
      if (node->count == 0)
        // this is a inner node - exit
        return;

      box3f bounds; bounds.set_empty();
      for (int i=0;i<node->count;i++) {
        const box3f primBox = boxes[bvh.primIDs[node->offset+i]];
        bounds.lower = min(bounds.lower,primBox.lower);
        bounds.upper = max(bounds.upper,primBox.upper);
      }
        
      int parentID = (refitData[nodeID] >> 1);
      while (true) {
        node->bounds = bounds;
        __threadfence();
        if (node == bvh.nodes)
          break;

        uint32_t refitBits = atomicAdd(&refitData[parentID],1u);
        if ((refitBits & 1) == 0)
          // we're the first one - let other one do it
          break;
        
        nodeID   = parentID;
        node     = &bvh.nodes[parentID];
        parentID = (refitBits >> 1);
        
        BinaryBVH::Node l = bvh.nodes[node->offset+0];
        BinaryBVH::Node r = bvh.nodes[node->offset+1];
        bounds.lower = min(l.bounds.lower,r.bounds.lower);
        bounds.upper = max(l.bounds.upper,r.bounds.upper);
      }
    }

    void refit(BinaryBVH &bvh,
               const box3f *boxes,
               cudaStream_t s=0)
    {
      uint32_t *refitData;
      CUBQL_CUDA_CALL(MallocAsync((void**)&refitData,bvh.numNodes*sizeof(int),s));
      int numNodes = bvh.numNodes;
      refit_init<<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,refitData,bvh.numNodes);
      refit_run<<<divRoundUp(numNodes,32),32,0,s>>>
        (bvh,refitData,boxes);
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      CUBQL_CUDA_CALL(FreeAsync((void*)refitData,s));
    }

    __global__
    void computeNodeCosts(BinaryBVH bvh, float *nodeCosts)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= bvh.numNodes) return;

      if (nodeID == 1) { nodeCosts[nodeID] = 0.f; return; }

      auto node = bvh.nodes[nodeID];
      float area = surfaceArea(node.bounds) / surfaceArea(bvh.nodes[0].bounds);
      if (node.count == 0)
        nodeCosts[nodeID] = area;
      else 
        nodeCosts[nodeID] = area * node.count;
    }
    
    float computeSAH(const BinaryBVH &bvh)
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
} // ::cuBQL

  // void gpuBuilder(BinaryBVH   &bvh,
  //                 const box3f *boxes,
  //                 uint32_t     numBoxes,
  //                 BuildConfig  buildConfig,
  //                 cudaStream_t s)
  // {
  //   if (buildConfig.makeLeafThreshold == 0)
  //     // unless explicitly specified, use default for spatial median
  //     // builder:
  //     buildConfig.makeLeafThreshold = 8;
  //   gpuBuilder_impl::build(bvh,boxes,numBoxes,buildConfig,s);
  //   gpuBuilder_impl::refit(bvh,boxes,s);
  //   CUBQL_CUDA_CALL(StreamSynchronize(s));
  // }

  // float computeSAH(const BinaryBVH &bvh)
  // {
  //   return gpuBuilder_impl::computeSAH(bvh);
  // }
  
  // void free(BinaryBVH   &bvh,
  //           cudaStream_t s)
  // {
  //   CUBQL_CUDA_CALL(StreamSynchronize(s));
  //   CUBQL_CUDA_CALL(FreeAsync(bvh.primIDs,s));
  //   CUBQL_CUDA_CALL(FreeAsync(bvh.nodes,s));
  //   CUBQL_CUDA_CALL(StreamSynchronize(s));
  //   bvh.primIDs = 0;
  // }
// }

