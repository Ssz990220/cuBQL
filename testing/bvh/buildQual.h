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

/*! \file buildBench.h Implements simple benchmark app for bvh
  construction */
#pragma once

// #define CUBQL_GPU_BUILDER_IMPLEMENTATION

#include "cuBQL/bvh.h"
#include "samples/common/CmdLine.h"
#include "samples/common/IO.h"
#include "testing/common/testRig.h"
#include "samples/common/Generator.h"
#include <set>
#include "cuBQL/traversal/shrinkingRadiusQuery.h"
#include <cuda.h>

namespace testing {

  typedef enum {
    BUILDTYPE_DEFAULT=0,
    BUILDTYPE_RADIX,
    BUILDTYPE_REBIN,
    BUILDTYPE_SAH }
    BuildType;
  using namespace cuBQL;
  using namespace cuBQL::samples;
      
  using vecND   = cuBQL::vec_t<double,CUBQL_TEST_D>;
  using vec_t   = cuBQL::vec_t<CUBQL_TEST_T,CUBQL_TEST_D>;
  using box_t   = cuBQL::box_t<CUBQL_TEST_T,CUBQL_TEST_D>;
  using point_t = cuBQL::vec_t<CUBQL_TEST_T,CUBQL_TEST_D>;
  using bvh_t   = cuBQL::bvh_t<CUBQL_TEST_T,CUBQL_TEST_D>;

  void computeBoxes(box_t *d_boxes,
                    const point_t *d_data,
                    int numData);
  bvh_t computeBVH(const box_t *d_boxes,
                   int numBoxes,
                   BuildType buildType);
  void free(bvh_t bvh);

  void usage(const std::string &error = "")
  {
    if (!error.empty())
      std::cout << "Error: " << error << "\n\n";
    std::cout << "Usage: ./cuBQL...buildBench... -n <numPoints> [--clustered|-c] [--uniform|-u]" << std::endl;
    exit(error.empty()?0:1);
  }

  __global__ void runQueries(uint64_t *d_numNodesVisited,
                             uint64_t *d_numPrimsVisited,
                             bvh_t bvh,
                             const vec_t *points,
                             int numPoints)
  {
    // using box_t   = cuBQL::box_t<T,D>;
    // using vec_t   = cuBQL::vec_t<T,D>;
    // using bvh_t   = cuBQL::bvh_t<T,D>;
    using node_t  = typename bvh_t::Node;
    
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPoints) return;

    vec_t queryPoint = points[tid];
    uint64_t numNodesVisited = 0;
    uint64_t numPrimsVisited = 0;
    auto nodeDist = [&](const node_t &node) -> float
    {
      numNodesVisited++;
      return fSqrDistance_rd(queryPoint,node.bounds);
    };
    auto primCode = [&](uint32_t primID) {
      numPrimsVisited++;
      vec_t point = points[primID];
      if (primID == tid) return INFINITY;
      return fSqrDistance_rd(queryPoint,point);
    };
    shrinkingRadiusQuery::forEachPrim(primCode,nodeDist,bvh);
    atomicAdd((unsigned long long int *)d_numNodesVisited,
              (unsigned long long int)numNodesVisited);
    atomicAdd((unsigned long long int *)d_numPrimsVisited,
              (unsigned long long int)numPrimsVisited);
  }
                             

  void runQuery(bvh_t bvh, const vec_t *points, int numPoints)
  {
    uint64_t *p_numNodesVisited = 0;
    uint64_t *p_numPrimsVisited = 0;
    CUBQL_CUDA_CALL(Malloc((void **)&p_numNodesVisited,sizeof(uint64_t)));
    CUBQL_CUDA_CALL(Malloc((void **)&p_numPrimsVisited,sizeof(uint64_t)));
    CUBQL_CUDA_CALL(Memset(p_numNodesVisited,0,sizeof(uint64_t)));
    CUBQL_CUDA_CALL(Memset(p_numPrimsVisited,0,sizeof(uint64_t)));
    runQueries
      <<<divRoundUp(numPoints,128),128>>>
      (p_numNodesVisited,
       p_numPrimsVisited,
       bvh,points,numPoints);
    uint64_t numNodesVisited;
    uint64_t numPrimsVisited;
    CUBQL_CUDA_CALL(Memcpy(&numNodesVisited,p_numNodesVisited,sizeof(numNodesVisited),
                           cudaMemcpyDefault));
    CUBQL_CUDA_CALL(Memcpy(&numPrimsVisited,p_numPrimsVisited,sizeof(numPrimsVisited),
                           cudaMemcpyDefault));
    printf("  NUM_TRAVERSAL_STEPS %s\n",
           prettyNumber(numNodesVisited+numPrimsVisited).c_str());
    // std::cout << " --> num NODES visited " << numNodesVisited << std::endl;
    // std::cout << " --> num PRIMS visited " << numPrimsVisited << std::endl;
    CUBQL_CUDA_CALL(Free(p_numNodesVisited));
    CUBQL_CUDA_CALL(Free(p_numPrimsVisited));
  }
                  
  
  void main(int ac, char **av,
            cuBQL::testRig::DeviceAbstraction &device)
  {
    CmdLine     cmdLine(ac,av);
    BuildType   buildType = BUILDTYPE_DEFAULT;
    std::string inFileName;
    while (!cmdLine.consumed()) {
      const std::string arg = cmdLine.getString();
      if (arg == "--radix" || arg == "--morton")
        buildType = BUILDTYPE_RADIX;
      else if (arg == "--rebin")
        buildType = BUILDTYPE_REBIN;
      else if (arg == "--sah")
        buildType = BUILDTYPE_SAH;
      else if (arg[0] != '-')
        inFileName = arg;
      else 
        usage("un-recognized cmd-line argument '"+arg+"'");
    }

    std::vector<vecND> generatedPoints = loadBinary<vecND>(inFileName);
    std::vector<point_t> dataPoints
      = cuBQL::samples::convert<CUBQL_TEST_T>(generatedPoints);
    
    const point_t *d_dataPoints
      = device.upload(dataPoints);
    int numData = int(dataPoints.size());
        
    std::cout << "computing boxes for bvh build" << std::flush << std::endl;
    int numBoxes = numData;
    box_t *d_boxes = device.alloc<box_t>(numBoxes);
    computeBoxes(d_boxes,d_dataPoints,numData);
        
    std::cout << "computing bvh" << std::flush << std::endl;
    bvh_t bvh = computeBVH(d_boxes,numBoxes,buildType);

    std::cout << "running closest-point queries, one query per data points, to measure how good the just built bvh actually is" << std::flush << std::endl;
    runQuery(bvh,d_dataPoints,numData);
    free(bvh);
    
    device.free(d_dataPoints);
  }

} // ::testing

