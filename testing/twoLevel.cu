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

#include "cuBQL/bvh.h"
#include "cuBQL/queries/fcp.h"

#include "testing/helper/CUDAArray.h"
#include "testing/helper.h"

namespace cuBQL {
  template<typename bvh_t>
  struct BLAS {
    bvh_t   bvh;
    float3 *data;
  };
  
  template<typename bvh_t>
  inline __device__
  int2 twoLevel_fcp(bvh_t bvh,
                    const BLAS<bvh_t> *blases,
                    float3 query,
                    float &maxQueryDistSquare);

  /*! fcp kernel for a binary two-level BVH. The 'blases' are a list
      of the bottom-level acceleration structures (also BinaryBVHs),
      maxQueryDist isthe maximum search distance. Return value is a
      int2, where result.x is the index of the BLAS that contained the
      closest point, and index.y is the ID of the primitive within
      that BLAS. If no point could be found within the given search
      radius, this returns result.x==-1 */
  inline __device__
  int2 twoLevel_fcp(BinaryBVH<float,3> bvh,
                   const BLAS<BinaryBVH<float,3>> *blases,
                   float3 query,
                   float &maxQueryDistSquare)
  {
    int2 result = {-1,-1};
    
    int2 stackBase[32], *stackPtr = stackBase;
    int nodeID = 0;
    int offset = 0;
    int count  = 0;
    while (true) {
      while (true) {
        offset = bvh.nodes[nodeID].offset;
        count  = bvh.nodes[nodeID].count;
        if (count>0)
          // leaf
          break;
        BinaryBVH<float,3>::Node child0 = bvh.nodes[offset+0];
        BinaryBVH<float,3>::Node child1 = bvh.nodes[offset+1];
        float dist0 = sqrDistance(child0,query);
        float dist1 = sqrDistance(child1,query);
        int closeChild = offset + ((dist0 > dist1) ? 1 : 0);
        if (dist1 <= maxQueryDistSquare) {
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
      int offset = bvh.nodes[nodeID].offset;
      int count  = bvh.nodes[nodeID].count;
      for (int i=0;i<count;i++) {
        int blasID = bvh.primIDs[offset+i];
        auto blas = blases[blasID];
        int blasResult = fcp(blas.bvh,blas.data,query,maxQueryDistSquare);
        if (blasResult > -1) {
          result.x = blasID;
          result.y = blasResult;
        }
      }
      while (true) {
        if (stackPtr == stackBase)
          return result;
        --stackPtr;
        if (__int_as_float(stackPtr->y) > maxQueryDistSquare) continue;
        nodeID = stackPtr->x;
        break;
      }
    }
  }

  /*! fcp kernel for a binary two-level BVH. The 'blases' are a list
    of the bottom-level acceleration structures (also BinaryBVHs),
    maxQueryDist isthe maximum search distance. Return value is a
    int2, where result.x is the index of the BLAS that contained the
    closest point, and index.y is the ID of the primitive within
    that BLAS. If no point could be found within the given search
    radius, this returns result.x==-1 */
  template<int N>
  inline __device__
  int2 twoLevel_fcp(WideBVH<float,3,N> bvh,
                    const BLAS<WideBVH<float,3,N>> *blases,
                    float3 query,
                    float &maxQueryDistSquare)
  {
    int2 result = {-1,-1};

    enum { stackSize = 64 };
    uint64_t stackBase[stackSize], *stackPtr = stackBase;
    int nodeID = 0;
    ChildOrder<N> childOrder;
    while (true) {
      while (true) {
        while (nodeID == -1) {
          if (stackPtr == stackBase)
            return result;
          uint64_t tos = *--stackPtr;
          if (__int_as_float(tos>>32) > maxQueryDistSquare)
            continue;
          nodeID = (uint32_t)tos;
          // pop....
        }
        if (nodeID & (1<<31))
          break;
        
        const typename WideBVH<float,3,N>::Node &node = bvh.nodes[nodeID];
#pragma unroll(N)
        for (int c=0;c<N;c++) {
          const auto child = node.children[c];
          if (!node.children[c].valid)
            childOrder.clear(c);
          else {
            float dist2 = sqrDistance(child.bounds,query);
            if (dist2 > maxQueryDistSquare) 
              childOrder.clear(c);
            else {
              uint32_t payload
                = child.count
                ? ((1<<31)|(nodeID<<log_of<N>::value)|c)
                : child.offset;
              childOrder.set(c,dist2,payload);
            }
          }
        }
        sort(childOrder);
#pragma unroll
        for (int c=N-1;c>0;--c) {
          uint64_t coc = childOrder.v[c];
          if (coc != uint64_t(-1)) {
            *stackPtr++ = coc;
          }
        }
        if (childOrder.v[0] == uint64_t(-1)) {
          nodeID = -1;
          continue;
        }
        nodeID = uint32_t(childOrder.v[0]);
      }
      
      int c = nodeID & ((1<<log_of<N>::value)-1);
      int n = (nodeID & 0x7fffffff)  >> log_of<N>::value;
      int offset = bvh.nodes[n].children[c].offset;
      int count  = bvh.nodes[n].children[c].count;
      for (int i=0;i<count;i++) {
        int blasID = bvh.primIDs[offset+i];
        auto blas = blases[blasID];
        int blasResult = fcp(blas.bvh,blas.data,query,maxQueryDistSquare);
        if (blasResult > -1) {
          result.x = blasID;
          result.y = blasResult;
        }
      }
      nodeID = -1;
    }
  }


}

namespace testing {

  using box_t = cuBQL::box3f;

  std::vector<float> reference;
  
  struct TestConfig {
    float maxTimeThreshold = 10.f;
    float maxQueryRadius = INFINITY;
  };
  
  void usage(const std::string &error = "")
  {
    if (!error.empty()) {
      std::cerr << error << "\n\n";
    }
    std::cout << "./cuBQL_fcp dataPoints.dat queryPoints.dat\n\n";
    exit(error.empty()?0:1);
  }

  __global__
  void makeBoxes(box_t *boxes, float3 *points, int numPoints)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;
    float3 point = points[tid];
    boxes[tid].lower = point;
    boxes[tid].upper = point;
  }

  __global__
  void resetResults(float *results, int numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;
    results[tid] = INFINITY;//-1;
  }
  
  template<typename bvh_t>
  __global__
  void runFCP(float *results,
              bvh_t bvh,
              const BLAS<bvh_t> *blases,
              const float3 *queries,
              float maxQueryRadius,
              int   numQueries)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numQueries) return;

    const float3 query = queries[tid];
    /* careful: fcp kernel expects SQUARE of maxradius */
    float sqrMaxDist = maxQueryRadius*maxQueryRadius;
    int2 res = cuBQL::twoLevel_fcp(bvh,blases,query,
                                   /* this also serves as return value: */
                                   sqrMaxDist);
    if (res.x == -1)
      results[tid] = INFINITY;
    else
      // results[tid] = sqrDistance(blases[res.x].data[res.y],query);
      results[tid] = sqrMaxDist;
  }

  template<typename bvh_t>
  void testFCP(const std::vector<std::vector<float3>> &h_dataPoints,
               const std::vector<float3> &h_queryPoints,
               BuildConfig buildConfig,
               TestConfig testConfig
               )
  {
    std::vector<BLAS<bvh_t>> blases(h_dataPoints.size());
    std::vector<box3f> blasBoxes(h_dataPoints.size());
    std::vector<CUDAArray<float3>> blasDatas(h_dataPoints.size());
    for (int blasID=0;blasID<h_dataPoints.size();blasID++) {
      CUDAArray<float3> &thisBlasData = blasDatas[blasID];
      thisBlasData.upload(h_dataPoints[blasID]);
      
      box3f bbox; bbox.set_empty();
      for (auto pt : h_dataPoints[blasID])
        bbox.grow(pt);
      blasBoxes[blasID] = bbox;
      
      CUDAArray<box_t> boxes(thisBlasData.size());
      {
        int bs = 256;
        int nb = divRoundUp((int)thisBlasData.size(),bs);
        makeBoxes<<<nb,bs>>>(boxes.data(),thisBlasData.data(),(int)thisBlasData.size());
      };
      CUBQL_CUDA_SYNC_CHECK();
      
      // cuBQL::BinaryBVH
      bvh_t &bvh = blases[blasID].bvh;
      blases[blasID].data = thisBlasData.data();
      std::cout << "building blas over " << prettyNumber(h_dataPoints.size()) << " points..." << std::endl;
      cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),buildConfig);
      std::cout << "blas bvh is built, SAH cost is " << cuBQL::computeSAH(bvh) << std::endl;
    }

    CUDAArray<box3f> d_blasBoxes;
    d_blasBoxes.upload(blasBoxes);
    CUDAArray<BLAS<bvh_t>> d_blases;
    d_blases.upload(blases);

    bvh_t tlas;
    buildConfig.maxAllowedLeafSize = 1;
    cuBQL::gpuBuilder(tlas,d_blasBoxes.data(),d_blasBoxes.size(),buildConfig);
    std::cout << "tlas bvh is built, SAH cost is " << cuBQL::computeSAH(tlas) << std::endl;

    
    CUDAArray<float3> queryPoints;
    queryPoints.upload(h_queryPoints);
    
    int numQueries = queryPoints.size();
    CUDAArray<float> closest(numQueries);

    // ------------------------------------------------------------------
    std::cout << "first query for warm-up and checking reference data (if provided)"
              << std::endl;
    // ------------------------------------------------------------------
    resetResults<<<divRoundUp(numQueries,128),128>>>(closest.data(),numQueries);
    runFCP<<<divRoundUp(numQueries,128),128>>>
      (closest.data(),
       tlas,
       d_blases.data(),
       queryPoints.data(),
       testConfig.maxQueryRadius,
       numQueries);
    CUBQL_CUDA_SYNC_CHECK();
    if (reference.empty()) {
      reference = closest.download();
    } else {
      std::cout << "checking reference...." << std::endl;
      std::vector<float> ours = closest.download();
      if (ours.size() != reference.size())
        throw std::runtime_error("reference file size odes not have expected number of entries!?");
      for (int i=0;i<ours.size();i++) {
        if (ours[i] > reference[i] && sqr(reference[i]) <= sqr(testConfig.maxQueryRadius)) {
          std::cout << "ours/reference mismatch at index "
                    << i << "/" << reference.size() << ":" << std::endl;
          std::cout << "  ours      is " << ours[i] << std::endl;
          std::cout << "  reference is " << reference[i] << std::endl;
          throw std::runtime_error("does NOT match!");
        }
      }
      std::cout << "all good, ours matches reference array ..." << std::endl;
    }

    // ------------------------------------------------------------------
    // actual timing runs
    // ------------------------------------------------------------------
    int numPerRun = 1;
    while (true) {
      CUBQL_CUDA_SYNC_CHECK();
      std::cout << "timing run with " << numPerRun << " repetition.." << std::endl;
      double t0 = getCurrentTime();
      for (int i=0;i<numPerRun;i++) {
        resetResults<<<divRoundUp(numQueries,128),128>>>(closest.data(),numQueries);
        runFCP<<<divRoundUp(numQueries,128),128>>>
          (closest.data(),
           tlas,
           d_blases.data(),
           queryPoints.data(),
           testConfig.maxQueryRadius,
           numQueries);
        CUBQL_CUDA_SYNC_CHECK();
      }
      double t1 = getCurrentTime();
      std::cout << "done " << numPerRun
                << " queries in " << prettyDouble(t1-t0) << "s, that's "
                << prettyDouble((t1-t0)/numPerRun) << "s query" << std::endl;
      if ((t1 - t0) > testConfig.maxTimeThreshold)
        break;
      numPerRun*=2;
    };
  }
}

using namespace testing;

int main(int ac, char **av)
{
  BuildConfig buildConfig;
  std::vector<std::string> dataFiles;
  std::string queryFile;
  std::string bvhType = "binary";
  bool make_reference = false;
  std::string referenceFileName;
  TestConfig testConfig;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (av[i][0] != '-')
      dataFiles.push_back(arg);
    else if (arg == "-q" || arg == "--query-file")
      queryFile = av[++i];
    else if (arg == "-bt" || arg == "--bvh-type")
      bvhType = av[++i];
    else if (arg == "-sah" || arg == "--sah")
      buildConfig.enableSAH();
    else if (arg == "-mr" || arg == "-mqr" || arg == "--max-query-radius")
      testConfig.maxQueryRadius = std::stof(av[++i]);
    else if (arg == "--check-reference") {
      referenceFileName = av[++i];
      make_reference = false;
    } else if (arg == "--make-reference") {
      referenceFileName = av[++i];
      make_reference = true;
      testConfig.maxTimeThreshold = 0.f;
    } else if (arg == "-mlt" || arg == "-lt")
      buildConfig.makeLeafThreshold = std::stoi(av[++i]);
    else
      usage("unknown cmd-line argument '"+arg+"'");
  }
  if (dataFiles.size() != 1)
    usage("unexpected number of data file names");
  std::vector<std::vector<float3>> dataBlocks
    = read_from<std::vector<std::vector<float3>>>(dataFiles[0]);
  for (auto df : dataFiles)
    dataBlocks.push_back(loadData<float3>(df));
  std::vector<float3> queryPoints = loadData<float3>(queryFile);
  
  if (!referenceFileName.empty() && !make_reference)
    reference = loadData<float>(referenceFileName);

  if (bvhType == "binary")
    testing::testFCP<BinaryBVH<float,3>>(dataBlocks,queryPoints,buildConfig,testConfig);
  // else if (bvhType == "bvh2")
  //   testing::testFCP<WideBVH<2>>(dataBlocks,queryPoints,buildConfig,testConfig);
  else if (bvhType == "bvh4")
    testing::testFCP<WideBVH<float,3,4>>(dataBlocks,queryPoints,buildConfig,testConfig);
  else if (bvhType == "bvh8")
    testing::testFCP<WideBVH<float,3,8>>(dataBlocks,queryPoints,buildConfig,testConfig);
  else
    throw std::runtime_error("unknown or not-yet-hooked-up bvh type '"+bvhType+"'");

  if (make_reference) {
    std::cout << "saving reference data to " << referenceFileName << std::endl;
    saveData(reference,referenceFileName);
  }
  return 0;
}
