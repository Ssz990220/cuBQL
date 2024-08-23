// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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
#include "cuBQL/builder/common.h"
#if CUBQL_HOST_BUILDER_IMPLEMENTATION
#include <vector>
#endif

namespace cuBQL {
  namespace host {
    // ******************************************************************
    // INTERFACE
    // (which functionality this header file provides)
    // ******************************************************************

    /*! a simple (and currently non parallel) recursive spatial median
        builder */
    template<typename T, int D>
    void spatialMedian(BinaryBVH<T,D>   &bvh,
                       const box_t<T,D> *boxes,
                       int                   numPrims,
                       BuildConfig           buildConfig);

    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************
    
#if CUBQL_HOST_BUILDER_IMPLEMENTATION
    namespace spatialMedian_impl {
      struct Topo {
        int offset;
        int count;
      };

      void makeLeaf(int nodeID, int begin, int end,
                    std::vector<Topo> &topo)
      {
        auto &node = topo[nodeID];
        node.count = end-begin;
        node.offset = begin;
      }
      
      int makeInner(int nodeID,
                    std::vector<Topo> &topo)
      {
        int childID = (int)topo.size();
        topo.push_back({});
        topo.push_back({});
        auto &node = topo[nodeID];
        node.count = 0;
        node.offset = childID;
        return childID;
      }
      
      template<typename T, int D>
      void buildRec(int nodeID, int begin, int end,
                    std::vector<Topo> &topo,
                    std::vector<int>  &primIDs,
                    std::vector<int>  &altPrimIDs,
                    const box_t<T,D>  *boxes,
                    BuildConfig        buildConfig)
      {
        if (end-begin <= buildConfig.makeLeafThreshold)
          return makeLeaf(nodeID,begin,end,topo);
        
        using box_t = ::cuBQL::box_t<T,D>;

        box_t centBounds;
        for (int i=begin;i<end;i++)
          centBounds.extend(boxes[primIDs[i]].center());
        if (centBounds.lower == centBounds.upper)
          return makeLeaf(nodeID,begin,end,topo);

        int dim = arg_max(centBounds.size());
        T   pos = centBounds.center()[dim];
        int Nl = 0, Nr = 0;
        for (int i=begin;i<end;i++) {
          int primID = primIDs[i];
          if (boxes[primID].center()[dim] < pos) {
            altPrimIDs[begin + Nl++] = primID;
          } else {
            altPrimIDs[end - ++Nr]   = primID;
          }
        }
        if (Nl == 0 || Nr == 0)
          return makeLeaf(nodeID,begin,end,topo);

        int mid = begin+Nl;
        for (int i=begin;i<end;i++)
          primIDs[i] = altPrimIDs[i];
        
        int childID = makeInner(nodeID,topo);
        buildRec(childID+0,begin,mid,topo,primIDs,altPrimIDs,boxes);
        buildRec(childID+1,mid,  end,topo,primIDs,altPrimIDs,boxes);
      }
      
      template<typename T, int D>
      void refit(int nodeID,
                 BinaryBVH<T,D>   &bvh,
                 const box_t<T,D> *boxes)
      {
        auto &node = bvh.nodes[nodeID];
        if (node.count == 0) {
          refit(node.offset+0,bvh,boxes);
          refit(node.offset+1,bvh,boxes);
          node.bounds = box_t<T,D>()
            .including(bvh.nodes[node.offset+0])
            .including(bvh.nodes[node.offset+1]);
        } else {
          node.bounds.clear();
          for (int i=0;i<node.count;i++)
            node.bounds.extend(boxes[bvh.primIDs[node.offset+i]]);
        }
      }
                         
      template<typename T, int D>
      void spatialMedian(BinaryBVH<T,D>   &bvh,
                         const box_t<T,D> *boxes,
                         int                   numPrims,
                         BuildConfig           buildConfig)
      {
        using box_t = ::cuBQL::box_t<T,D>;
        std::vector<int> primIDs;
        for (int i=0;i<numPrims;i++) {
          box_t box = boxes[i];
          if (box.empty()) continue;
          primIDs.push_back(i);
        }
        std::vector<int>  altPrimIDs(primIDs.size());
        std::vector<Topo> topo(1);
        
        buildRec(0,0,primIDs.size(),
                 topo,primIDs,altPrimIDs,boxes,buildConfig);
        altPrimIDs.clear();
        bvh.primIDs = new int[primIDs.size()];
        std::copy(primIDs.begin(),primIDs.end(),bvh.primIDs);
        primIDs.clear();

        bvh.nodes = new typename BinaryBVH<T,D>::Node[topo.size()];
        for (int i=0;i<(int)topo.size();i++) {
          bvh.nodes[i].admin.count = topo[i].count;
          bvh.nodes[i].admin.offset = topo[i].offset;
        }
        topo.clear();
        refit(0,bvh,boxes);
      }
    }
    
    /*! a simple (and currently non parallel) recursive spatial median
      builder */
    template<typename T, int D>
    void spatialMedian(BinaryBVH<T,D>   &bvh,
                       const box_t<T,D> *boxes,
                       int                   numPrims,
                       BuildConfig           buildConfig)
    {
      // spatialMedian_impl::build(bvh,boxes,numPrims,buildConfig);
    }

#endif
  }
}

#define CUBQL_HOST_INSTANTIATE_BINARY_BVH(T,D)                      \
  namespace cuBQL {                                                 \
    namespace host {                                                \
      template void spatialMedian(BinaryBVH<T,D>   &bvh,            \
                                  const box_t<T,D> *boxes,          \
                                  int               numPrims,       \
                                  BuildConfig       buildConfig);   \
    }                                                               \
  }                                                                 \
  

