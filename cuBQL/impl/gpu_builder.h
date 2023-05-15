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

namespace cubql {

  /*! builds a wide-bvh over a given set of primitmive bounding boxes.

    builder runs on the GPU; boxes[] must be a device-readable array
    (managed or device mem); bvh arrays will be allocated in device mem 

    primitives may be marked as "inactive/invalid" by using a bounding
    box whose lower/upper coordinates are inverted; such primitmives
    will be ignored, and will thus neither be visited during traversal
    nor mess up the tree in any way, shape, or form
  */
  void gpuBuilder(BinaryBVH &bvh,
                  const box3f *boxes,
                  uint32_t numBoxes,
                  int maxLeafSize);
  
#if CUBQL_GPU_BUILDER_IMPLEMENTATION
  
  namespace gpuBuilder_impl {

    struct PrimState {
      int      nodeID; //!< node the given prim is (currently) in.
      uint32_t done  : 1;
      uint32_t primID:31; //!< prim we're talking about
    };

    typedef enum { OPEN_BRANCH, OPEN_NODE, DONE_NODE } NodeState;
    
    template<typename box_t>
    struct AtomicBox {
      inline __device__ void set_empty();
      inline __device__ void atomic_grow(const box_t &box);
      inline __device__ float get_center(int dim) const
      { return 0.5f*(decode(lower[dim])+decode(upper[dim])); }
      inline __device__ box_t make_box() const;
      
      uint32_t lower[box_t::numDims];
      uint32_t upper[box_t::numDims];

    private:
      inline static __device__ uint32_t encode(float f);
      inline static __device__ float    decode(uint32_t ui);
      
    };
      
    template<typename box_t>
    inline __device__ box_t AtomicBox<box_t>::make_box() const {
      box_t box;
      for (int d=0;d<box_t::numDims;d++) {
        box.lower[d] = decode(lower[d]);
        box.upper[d] = decode(upper[d]);
      }
      return box;
    }
    
    template<typename box_t>
    inline __device__ uint32_t AtomicBox<box_t>::encode(float f)
    {
      const int32_t sign = 0x80000000;
      int32_t bits = __float_as_int(f);
      if (bits & sign) bits ^= 0x7fffffff;
      return bits;
    }
      
    template<typename box_t>
    inline __device__ float AtomicBox<box_t>::decode(uint32_t bits)
    {
      const uint32_t sign = 0x80000000;
      if (bits & sign) bits ^= 0x7fffffff;
      return __int_as_float(bits);
    }
    
    template<typename box_t>
    inline __device__ void AtomicBox<box_t>::set_empty()
    {
      for (int d=0;d<box_t::numDims;d++) {
        lower[d] = encode(+INFINITY);
        upper[d] = encode(-INFINITY);
      }
    }
    template<typename box_t>
    inline __device__ void AtomicBox<box_t>::atomic_grow(const box_t &other)
    {
      for (int d=0;d<box_t::numDims;d++) {
        atomicMin(&lower[d],encode(other.get_lower(d)));
        atomicMax(&upper[d],encode(other.get_upper(d)));
      }
    }
    
    struct TempNode {
      union {
        struct {
          AtomicBox<box3f> centBounds;
          uint32_t         count;
        } openBranch;
        struct {
          uint32_t offset;
          int      dim;
          union {
            uint32_t tieBreaker;
            float    pos;
          };
        } openNode;
        struct {
          uint32_t offset;
          uint32_t count;
        } doneNode;
      };
    };
    
    struct BuildState {
      uint32_t  numNodes;
      TempNode *nodes;
    };

    __global__ void initState(BuildState *buildState,
                              NodeState  *nodeStates,
                              TempNode   *nodes)
    {
      buildState->nodes = nodes;
      buildState->numNodes = 2;
      nodeStates[0] = OPEN_BRANCH;
      nodes[0].openBranch.count = 0;
      nodes[0].openBranch.centBounds.set_empty();
    }
  
    __global__ void initPrims(TempNode    *nodes,
                              PrimState   *primState,
                              const box_t *primBoxes,
                              uint32_t     numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;
      
      auto &me = primState[primID];
      me.primID = primID;
                                                    
      const box_t box = primBoxes[primID];
      if (box.get_lower(0) <= box.get_upper(0)) {
        me.nodeID = 0;
        me.done   = false;
        atomicAdd(&nodes[0].openBranch.count,1);
        nodes[0].openBranch.centBounds.atomic_grow(box);
      } else {
        me.nodeID = -1;
        me.done   = true;
      }
    }

    __global__ void selectSplits(BuildState<box_t> *buildState,
                                 NodeState       *nodeStates,
                                 TempNode<box_t> *nodes,
                                 uint32_t         numNodes,
                                 int              maxLeafSize)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      NodeState &nodeState = nodeStates[nodeID];
      if (nodeState == DONE_NODE)
        // this node was already closed before
        return;
      if (nodeState == OPEN_NODE) {
        // this node was open in the last pass, can close it.
        nodeState = DONE_NODE;
        return;
      }
      
      auto in = nodes[nodeID].openBranch;
      if (in.count < maxLeafSize) {
        auto &done = nodes[nodeID].doneNode;
        done.count  = in.count;
        // set this to max-value, so the prims can later do atomicMin
        // with their position ion the leaf list; this value is
        // greater than any prim position.
        done.offset = (uint32_t)-1;
        nodeState = DONE_NODE;
      } else {
        float widestWidth = 0.f;
        int   widestDim   = -1;
        for (int d=0;d<box_t::numDims;d++) {
          float width = in.centBounds.upper[d] - in.centBounds.lower[d];
          if (width <= widestWidth)
            continue;
          widestWidth = width;
          widestDim   = d;
        }
      
        auto &open = nodes[nodeID].openNode;
        open.dim = widestDim;
        if (widestDim >= 0) 
          open.pos = in.centBounds.get_center(widestDim);
        open.offset = atomicAdd(&buildState->numNodes,2);
        for (int side=0;side<2;side++) {
          const int childID = open.offset+side;
          auto &child = nodes[childID].openBranch;
          child.centBounds.set_empty();
          child.count = 0;
          nodeStates[childID] = OPEN_BRANCH;
        }
      }
      nodeState = OPEN_NODE;
    }

    __global__
    void updatePrims(NodeState       *nodeStates,
                     TempNode<box_t> *nodes,
                     PrimState       *primStates,
                     const box_t     *primBoxes,
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
      const box_t primBox = primBoxes[me.primID];
      int side = 0;
      if (split.dim == -1) {
        side = atomicAdd(&split.tieBreaker,1) & 1;
      } else {
        const float center = 0.5f*(primBox.get_lower(split.dim)+
                                   primBox.get_upper(split.dim));
        side = (center >= split.pos);
      }
      auto &myBranch = nodes[split.offset+side].openBranch;
      atomicAdd(&myBranch.count,1);
      myBranch.centBounds.atomic_grow(primBox);
    }

    /* given a sorted list of {nodeID,primID} pairs, this kernel does
       two things: a) it extracts the 'primID's and puts them into the
       bvh's primIDs[] array; and b) it writes, for each leaf nod ein
       the nodes[] array, the node.offset value to point to the first
       of this nodes' items in that bvh.primIDs[] list. */
    __global__
    void writePrimsAndLeafOffsets(TempNode<box_t> *nodes,
                                  uint32_t        *bvhItemList,
                                  PrimState       *primStates,
                                  int numPrims)
    {
      const int offset = threadIdx.x+blockIdx.x*blockDim.x;
      if (offset >= numPrims) return;

      auto &ps = primStates[offset];
      bvhItemList[offset] = ps.primID;
      
      if ((int)ps.nodeID < 0)
        /* invalid prim, just skip here */
        return;
      auto &node = nodes[ps.nodeID];
      atomicMin(&node.offset,offset);
    }

    void build(BinaryBVH &bvh,
               const box3f *boxes,
               int numPrims,
               int maxLeafSize)
    {
      // ==================================================================
      // do build on temp nodes
      // ==================================================================
      CUDAArray<TempNode<box3f>>   tempNodes(2*numPrims);
      CUDAArray<NodeState>         nodeStates(2*numPrims);
      CUDAArray<PrimState>         primStates(numPrims);
      CUDAArray<BuildState<box3f>> buildState(1);
      initState<<<1,1>>>(buildState.data(),
                         nodeStates.data(),
                         tempNodes.data());
      initPrims<<<divRoundUp(numPrims,1024),1024>>>
        (tempNodes.data(),
         primStates.data(),boxes,numPrims);

      int numDone = 0;
      int numNodes;
      while (true) {
        CUBQL_CUDA_CALL(Memcpy(&numNodes,&buildState.data()->numNodes,
                               sizeof(numNodes),cudaMemcpyDeviceToHost));
        if (numNodes == numDone)
          break;
        
        selectSplits<<<divRoundUp(numNodes,1024),1024>>>
          (buildState.data(),
           nodeStates.data(),tempNodes.data(),numNodes,
           maxLeafSize);
        
        numDone = numNodes;
        
        updatePrims<<<divRoundUp(numPrims,1024),1024>>>
          (nodeStates.data(),tempNodes.data(),
           primStates.data(),boxes,numPrims);
      }

      // ==================================================================
      // sort {item,nodeID} list
      // ==================================================================
      
      void *d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      CUDAArray<PrimState> sortedPrimStates(numPrims);
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates.data(),
                                     (uint64_t*)sortedPrimStates.data(),
                                     numPrims,32,64);

      CUBQL_CUDA_CALL(Malloc(&d_temp_storage,temp_storage_bytes));
      cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates.data(),
                                     (uint64_t*)sortedPrimStates.data(),
                                     numPrims,32,64);
      CUBQL_CUDA_CALL(Free(d_temp_storage));
      primStates.free();

      // ==================================================================
      // allocate and write BVH item list, and write offsets of leaf nodes
      // ==================================================================

      bvh.numPrims = numPrims;
      CUBQL_CUDA_CALL(Malloc(&bvh.primIDs,numPrims*sizeof(int)));
      writePrimsAndLeafOffsets<<<divRoundUp(numPrims,1024),1024>>>
        (tempNodes.data(),bvh.primIDs,sortedPrimStates.data(),numPrims);
    }
  }
  
  void gpuBuilder(BinaryBVH   &bvh,
                  const box3f *boxes,
                  uint32_t     numBoxes,
                  int          maxLeafSize)
  { gpuBuilder_impl::build(bvh,boxes,numBoxes,maxLeafSize); }
#endif
}

