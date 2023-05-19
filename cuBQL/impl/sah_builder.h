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

#if CUBQL_GPU_BUILDER_IMPLEMENTATION

namespace cuBQL {
  namespace sahBuilder_impl {
    using gpuBuilder_impl::AtomicBox;
    using gpuBuilder_impl::PrimState;
    using gpuBuilder_impl::NodeState;
    using gpuBuilder_impl::BuildState;
    using gpuBuilder_impl::OPEN_NODE;
    using gpuBuilder_impl::DONE_NODE;
    using gpuBuilder_impl::OPEN_BRANCH;
    using gpuBuilder_impl::_ALLOC;
    using gpuBuilder_impl::_FREE;

    struct CUBQL_ALIGN(16) TempNode {
      union {
        struct {
          AtomicBox<box3f> centBounds;
          uint32_t         count;
          uint32_t         unused;
        } openBranch;
        struct {
          AtomicBox<box3f> centBounds;
          uint32_t offset;
          int8_t   dim;
          int8_t   bin;
        } openNode;
        struct {
          uint32_t offset;
          uint32_t count;
          uint32_t unused[2];
        } doneNode;
      };
    };
    
    struct SAHBins {
      enum { numBins = 8 };
      struct {
        struct CUBQL_ALIGN(16) {
          AtomicBox<box3f> bounds;
          int   count;
        } bins[numBins];
      } dims[3];
    };

    inline __device__
    void grow(box3f &tgt, box3f addtl)
    {
      tgt.lower = min(tgt.lower,addtl.lower);
      tgt.upper = max(tgt.upper,addtl.upper);
    }
    
    inline __device__
    void evaluateSAH(int &splitDim,
                     int &splitBin,
                     const SAHBins &sah)
    {
      float bestCost = INFINITY;
      
      float rAreas[sah.numBins];
      for (int d=0;d<3;d++) {
        box3f box; box.set_empty();
        int   rCount = 0;
        for (int b=sah.numBins-1;b>=0;--b) {
          auto bin = sah.dims[d].bins[b];
          grow(box,bin.bounds.make_box());
          rCount += bin.count;
          rAreas[b] = surfaceArea(box);
        }
        const float leafCost = rAreas[0] * rCount;
        if (leafCost < bestCost) {
          bestCost = leafCost;
          splitDim = -1;
        }
        box.set_empty();
        int lCount = 0;
        for (int b=0;b<sah.numBins;b++) {
          float rArea = rAreas[b];
          float lArea = surfaceArea(box);
          if (lCount>0 && rCount>0) {
            float cost = lArea*lCount+rArea*rCount;
            if (cost < bestCost) {
              bestCost = cost;
              splitDim = d;
              splitBin = b;
            }
          }
          auto bin = sah.dims[d].bins[b];
          grow(box,bin.bounds.make_box());
          lCount += bin.count;
          rCount -= bin.count;
        }
      }
    }
    
    __global__ void initState(BuildState *buildState,
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
    }

    
    __global__
    void binPrims(SAHBins          *sahBins,
                  int               sahNodeBegin,
                  int               sahNodeEnd,
                  TempNode         *nodes,
                  PrimState        *primState,
                  const box3f      *primBoxes,
                  uint32_t          numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;

      auto ps = primState[primID];
      if (ps.done) return;

      int nodeID = ps.nodeID;
      if (nodeID < sahNodeBegin || nodeID >= sahNodeEnd)
        return;

      const box3f primBox = primBoxes[primID];

      auto &sah = sahBins[nodeID-sahNodeBegin];
      box3f centBounds = nodes[nodeID].openBranch.centBounds.make_box();
#pragma unroll(3)
      for (int d=0;d<3;d++) {
        int bin = 0;
        float lo = centBounds.get_lower(d);
        float hi = centBounds.get_upper(d);
        if (hi > lo) {
          float prim_d = 0.5f*(primBox.get_lower(d)+primBox.get_upper(d));
          float rel
            = (prim_d - centBounds.get_lower(d))
            / (centBounds.get_upper(d)-centBounds.get_lower(d));
          bin = int(rel*SAHBins::numBins);
          bin = max(0,min(SAHBins::numBins-1,bin));
          // printf("prim %i in node %i, pos %f %f %f in cent %f %f %f - %f %f %f; dim %i: rel %f bin %i\n",
          //        primID,nodeID,
          //        primBox.lower.x,
          //        primBox.lower.y,
          //        primBox.lower.z,
          //        centBounds.lower.x,
          //        centBounds.lower.y,
          //        centBounds.lower.z,
          //        centBounds.upper.x,
          //        centBounds.upper.y,
          //        centBounds.upper.z,
          //        d,rel,bin);
        }
        auto &myBin = sah.dims[d].bins[bin];
        atomic_grow(myBin.bounds,primBox);
        atomicAdd(&myBin.count,1);
      }
    }
    
    __global__
    void closeOpenNodes(BuildState *buildState,
                        NodeState  *nodeStates,
                        TempNode   *nodes,
                        int numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes)
        return;
      
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
      // cannot be anything else...
    }
    
    __global__
    void selectSplits(BuildState *buildState,
                      SAHBins    *sahBins,
                      int         sahNodeBegin,
                      int         sahNodeEnd,
                      NodeState  *nodeStates,
                      TempNode   *nodes,
                      int         maxLeafSize)
    {
      const int nodeID = sahNodeBegin + threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1) return;
      
      if (nodeID < sahNodeBegin || nodeID >= sahNodeEnd) return;
      
      NodeState &nodeState = nodeStates[nodeID];
      auto in = nodes[nodeID].openBranch;
      auto &sah = sahBins[nodeID-sahNodeBegin];
      int   splitDim = -1;
      int   splitBin;
      if (in.count >= maxLeafSize) {
        evaluateSAH(splitDim,splitBin,sah);
        // printf("evaluated sah, result is dim %i bin %i\n",splitDim,splitBin);
      }
      if (splitDim < 0) {
        nodeState   = DONE_NODE;
        auto &done  = nodes[nodeID].doneNode;
        done.count  = in.count;
        // set this to max-value, so the prims can later do atomicMin
        // with their position ion the leaf list; this value is
        // greater than any prim position.
        done.offset = (uint32_t)-1;
        // printf("#ss node %i making leaf, count %i\n",nodeID,in.count);
      } else {
        nodeState = OPEN_NODE;
        
        auto &open      = nodes[nodeID].openNode;
        open.dim = splitDim;
        open.bin = splitBin;
        open.centBounds = in.centBounds;
        open.offset = atomicAdd(&buildState->numNodes,2);
#pragma unroll
        for (int side=0;side<2;side++) {
          const int childID = open.offset+side;
          auto &child = nodes[childID].openBranch;
          child.centBounds.set_empty();
          child.count         = 0;
          nodeStates[childID] = OPEN_BRANCH;
        }
        // printf("#ss node %i making inner, offset %i, dim %i bin %i\n",
        //        nodeID,open.offset,open.dim,open.bin);
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

      auto open = nodes[me.nodeID].openNode;
      const box3f primBox = primBoxes[me.primID];

      const int d = open.dim;
      float lo = open.centBounds.get_lower(d);
      float hi = open.centBounds.get_upper(d);
      
      float prim_d = 0.5f*(primBox.get_lower(d)+primBox.get_upper(d));
      float rel
        = (prim_d - lo)
        / (hi - lo);
      int prim_bin = int(rel*SAHBins::numBins);
      prim_bin = max(0,min(SAHBins::numBins-1,prim_bin));
      
      int side = (prim_bin >= open.bin);
      // printf("updateprim %i node %i state %i dim %i bin %i -> prim bin %i -> side %i\n",
      //        primID,me.nodeID,ns,open.dim,open.bin,
      //        prim_bin,side);
      int newNodeID = open.offset+side;
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

    __global__
    void clearBins(SAHBins *sahBins, int numActive)
    {
      const int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numActive) return;

      for (int d=0;d<3;d++)
        for (int b=0;b<SAHBins::numBins;b++) {
          auto &mine = sahBins[tid].dims[d].bins[b];
          mine.count = 0;
          mine.bounds.set_empty();
        }
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


    
    void sahBuilder(BinaryBVH  &bvh,
                    const box3f *boxes,
                    int          numPrims,
                    int          maxLeafSize,
                    cudaStream_t s)
    {
      // std::cout << "#######################################################" << std::endl;
      // ==================================================================
      // do build on temp nodes
      // ==================================================================
      TempNode   *tempNodes = 0;
      NodeState  *nodeStates = 0;
      PrimState  *primStates = 0;
      BuildState *buildState = 0;
      SAHBins    *sahBins    = 0;
      int maxActiveSAHs = 4+numPrims/(8*SAHBins::numBins);
      _ALLOC(tempNodes,2*numPrims,s);
      _ALLOC(nodeStates,2*numPrims,s);
      _ALLOC(primStates,numPrims,s);
      _ALLOC(buildState,1,s);
      _ALLOC(sahBins,maxActiveSAHs,s);
      
      // PING; CUBQL_CUDA_SYNC_CHECK();
      
      initState<<<1,1,0,s>>>(buildState,
                             nodeStates,
                             tempNodes);
      initPrims<<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,
         primStates,boxes,numPrims);
      // PING; CUBQL_CUDA_SYNC_CHECK();
      clearBins<<<1,32,0,s>>>
        (sahBins,1);
      // PING; CUBQL_CUDA_SYNC_CHECK();

      int numDone = 0;
      int numNodes = 0;

      while (true) {
        // CUBQL_CUDA_SYNC_CHECK();
        CUBQL_CUDA_CALL(MemcpyAsync(&numNodes,&buildState->numNodes,
                                    sizeof(numNodes),cudaMemcpyDeviceToHost,s));
        CUBQL_CUDA_CALL(StreamSynchronize(s));
        // std::cout << "----- done/nodes " << numDone << " / " << numNodes << std::endl;
        if (numNodes == numDone)
          break;

        // close all nodes that might still be open in last round
        // PING; CUBQL_CUDA_SYNC_CHECK();
        if (numDone > 0)
          closeOpenNodes<<<divRoundUp(numDone,1024),1024,0,s>>>
            (buildState,nodeStates,tempNodes,numDone);
        
        // PING; CUBQL_CUDA_SYNC_CHECK();
        const int openBegin = numDone;
        const int openEnd   = numNodes;
        for (int sahBegin=openBegin;sahBegin<openEnd;sahBegin+=maxActiveSAHs) {
          // CUBQL_CUDA_SYNC_CHECK();
          int sahEnd = std::min(sahBegin+maxActiveSAHs,openEnd);
          int numSAH = sahEnd-sahBegin;
          // std::cout << "RANGE " << sahBegin << " .. " << sahEnd << std::endl;
          // PING; CUBQL_CUDA_SYNC_CHECK();
          // std::cout << "----clearing bins ...." << std::endl;;
          clearBins<<<divRoundUp(numSAH,32),32,0,s>>>
            (sahBins,numSAH);
          // PING; CUBQL_CUDA_SYNC_CHECK();
          // std::cout << "----binning prims .... node range " << sahBegin << ".." << sahEnd << std::endl;;
          binPrims<<<divRoundUp(numPrims,128),128,0,s>>>
            (sahBins,sahBegin,sahEnd,
             tempNodes,
             primStates,boxes,numPrims);

          // CUBQL_CUDA_SYNC_CHECK();

          // PING; CUBQL_CUDA_SYNC_CHECK();
          // std::cout << "----selecting splits num = " << numSAH << " .... node range " << sahBegin << ".." << sahEnd << std::endl;;
          selectSplits<<<divRoundUp(numSAH,32),32,0,s>>>
            (buildState,
             sahBins,sahBegin,sahEnd,
             nodeStates,tempNodes,
             maxLeafSize);
          
          // PING; CUBQL_CUDA_SYNC_CHECK();
          // CUBQL_CUDA_SYNC_CHECK();
        }
        

        
        // (buildState.data(),
        //  nodeStates.data(),tempNodes.data(),numNodes,
        //  maxLeafSize);
        
        numDone = numNodes;

        // CUBQL_CUDA_SYNC_CHECK();
        // std::cout << "UPDATING -------------------------------------------------------" << std::endl;
        // std::cout << "updateprims, node range "  << numDone << " / " << numNodes << std::endl;
        // PING; CUBQL_CUDA_SYNC_CHECK();
        updatePrims<<<divRoundUp(numPrims,1024),1024,0,s>>>
          (nodeStates,tempNodes,
           primStates,boxes,numPrims);
        // CUBQL_CUDA_SYNC_CHECK();
        // (nodeStates.data(),tempNodes.data(),
        //  primStates.data(),boxes,numPrims);
      }
       // PING; CUBQL_CUDA_SYNC_CHECK();
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
      _FREE(sahBins,s);
      // PING; CUBQL_CUDA_SYNC_CHECK();

      // std::cout << "#######################################################" << std::endl;
    }

  } // ::cuBQL::sahBuilder_impl

  void gpuSAHBuilder(BinaryBVH   &bvh,
                     const box3f *boxes,
                     uint32_t     numBoxes,
                     int          maxLeafSize,
                     cudaStream_t s)
  {
    sahBuilder_impl::sahBuilder(bvh,boxes,numBoxes,maxLeafSize,s);
    gpuBuilder_impl::refit(bvh,boxes,s);
    CUBQL_CUDA_CALL(StreamSynchronize(s));
  }

} // :: cuBQL

#endif
