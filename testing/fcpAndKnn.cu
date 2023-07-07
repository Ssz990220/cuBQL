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

// #define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/computeSAH.h"
#include "cuBQL/queries/fcp.h"
#include "cuBQL/queries/knn.h"

#include "testing/helper/CUDAArray.h"
#include "testing/helper.h"
#include "testing/helper/Generator.h"
#include <fstream>
#include <sstream>

#define PLOT_BOXES 1

namespace cuBQL {

  namespace test_rig {

    std::vector<float> reference;
  
    struct TestConfig {
      /* knn_k = 0 means fcp, every other number means knn-query with this k */
      int knn_k = 0;
      float maxTimeThreshold = 10.f;
      float maxQueryRadius = INFINITY;

      std::string dataGen = "uniform";
      int dataCount = 100000;
      std::string queryGen = "uniform";
      int queryCount = 100000;

      bool make_reference = false;
      std::string referenceFileName;

      bool dumpTestData = false;
    };
  
    void usage(const std::string &error = "")
    {
      if (!error.empty()) {
        std::cerr << error << "\n\n";
      }
      std::cout << "./cuBQL_fcpAndKnn <args>\n\n";
      std::cout << "w/ args:\n";
      std::cout << "-dc <data_count>\n";
      std::cout << "-dg <data_generator_string> (see generator strings)\n";
      std::cout << "-qc <guery_count> (see generator strings)\n";
      std::cout << "-qg <query_generator_string> (see generator strings)\n";
      
      exit(error.empty()?0:1);
    }

#if USE_BOXES
#else
    template<int D>
    __global__
    void makeBoxes(cuBQL::box_t<float,D> *boxes,
                   cuBQL::vec_t<float,D> *points,
                   int numPoints)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numPoints) return;
      vec_t<float,D> point = points[tid];
      boxes[tid].lower = point;
      boxes[tid].upper = point;
    }
#endif

    __global__
    void resetResults(float *results, int numQueries)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numQueries) return;
      results[tid] = INFINITY;//-1;
    }


  
  
    template<typename T, int D>
    void plotBoxes(const CUDAArray<box_t<T,D>> &d_boxes)
    {
      if (getenv("CUBQL_PLOT_BOXES") == 0)
        return;
      std::cout << "plotting boxes..." << std::endl;
      std::vector<box_t<T,D>> boxes = d_boxes.download();
      for (int u=0;u<D;u++) {
        for (int v=u+1;v<D;v++) {
          std::stringstream fileName;
          fileName << "boxes_" << char('x'+u) << "_" << char('x'+v) << ".xfig";
          std::ofstream file(fileName.str());
          file << "#FIG 3.2  Produced by xfig version 3.2.7b" << std::endl;
          file << "Landscape" << std::endl;
          file << "Center" << std::endl;
          file << "Metric" << std::endl;
          file << "A4" << std::endl;
          file << "100.00" << std::endl;
          file << "Single" << std::endl;
          file << "-2" << std::endl;
          file << "1200 2" << std::endl;

          float scale = 10000.f;
          for (auto box : boxes) {
            int x0 = int(scale * box.lower[u]);
            int y0 = int(scale * box.lower[v]);
            int x1 = int(scale * box.upper[u]);
            int y1 = int(scale * box.upper[v]);
            int thick = 3;
            if (x1-x0 < thick) x1 = x0+thick;
            if (y1-y0 < thick) y1 = y0+thick;
            file << "2 2 0 " << thick << " 0 7 50 -1 -1 0.000 0 0 -1 0 0 5" << std::endl;
            file << "\t";
            file << " " << x0 << " " << y0;
            file << " " << x1 << " " << y0;
            file << " " << x1 << " " << y1;
            file << " " << x0 << " " << y1;
            file << " " << x0 << " " << y0;
            file << std::endl;
          }
        }
      }
    }
  
    // ------------------------------------------------------------------
    template<int D, typename bvh_t>
    __global__
    void runFCP(float        *results,
                bvh_t         bvh,
#if USE_BOXES
                const box_t<float,D>    *prims,
#else
                const vec_t<float,D>    *prims,
#endif
                const vec_t<float,D> *queries,
                float         maxRadius,
                int           numQueries
#if DO_STATS
                ,Stats *d_stats
#endif
                )
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numQueries) return;

      const vec_t<float,D> query = queries[tid];
      float sqrMaxQueryDist = maxRadius*maxRadius;
      int res = fcp(bvh,prims,query,
                    /* fcp kernel expects SQUARE of radius */
                    sqrMaxQueryDist
#if DO_STATS
                    ,d_stats
#endif
                    );
      if (res == -1)
        results[tid] = INFINITY;
      else
        // results[tid] = sqrDistance(dataPoints[res],query);
        results[tid] = sqrMaxQueryDist;
    }


    // ------------------------------------------------------------------
    template<int K, int D, typename bvh_t>
    __global__
    void runKNN(float        *results,
                bvh_t         bvh,
#if USE_BOXES
                const box_t<float,D>    *prims,
#else
                const vec_t<float,D>    *prims,
#endif
                const vec_t<float,D> *queries,
                float         maxRadius,
                int           numQueries
#if DO_STATS
                ,Stats *d_stats
#endif
                )
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numQueries) return;

      KNNResults<K> kNearest;
      kNearest.clear(maxRadius*maxRadius);
      const vec_t<float,D> query = queries[tid];
      knn(kNearest,bvh,prims,query
          /* fcp kernel expects SQUARE of radius */
#if DO_STATS
          ,d_stats
#endif
          );
      results[tid] = sqrtf(kNearest.maxDist2);
    }
  


    // ------------------------------------------------------------------
  
    template<int D, typename bvh_t=cuBQL::BinaryBVH<float,D>>
    void testFCP(TestConfig testConfig,
                 BuildConfig buildConfig)
    {
      using point_t = cuBQL::vec_t<float,D>;
      using box_t = cuBQL::box_t<float,D>;
      typename PointGenerator<float,D>::SP queryGenerator
        = PointGenerator<float,D>::createFromString(testConfig.queryGen);
      // std::vector<point_t> h_dataPoints;
      // std::vector<point_t> h_queryPoints;

      CUDAArray<point_t> queryPoints
        = queryGenerator->generate(testConfig.queryCount,0x23423498);
      if (testConfig.dumpTestData) {
        std::vector<point_t> h_queries = queryPoints.download();
        saveData(h_queries,"query_points");
      }
      
      // dataPoints.upload(h_dataPoints);

#if USE_BOXES
      typename BoxGenerator<float,D>::SP dataGenerator
        = BoxGenerator<float,D>::createFromString(testConfig.dataGen);
      CUDAArray<box_t> data
        = dataGenerator->generate(testConfig.dataCount,0x1345);
      auto &boxes = data;
#else
      typename PointGenerator<float,D>::SP dataGenerator
        = PointGenerator<float,D>::createFromString(testConfig.dataGen);
      CUDAArray<point_t> data
        = dataGenerator->generate(testConfig.dataCount,0x1345);

      if (testConfig.dumpTestData) {
        std::vector<point_t> h_data = data.download();
        saveData(h_data,"data_points");
      }
      
      CUDAArray<box_t> boxes(data.size());
      {
        int bs = 256;
        int nb = divRoundUp((int)data.size(),bs);
        makeBoxes<<<nb,bs>>>(boxes.data(),data.get(),(int)data.size());
      };
#endif


#if PLOT_BOXES
      plotBoxes(boxes);
#endif
    
      // cuBQL::BinaryBVH
      bvh_t bvh;
      cuBQL::gpuBuilder(bvh,boxes.data(),boxes.size(),buildConfig);
      if (D == 3) 
        std::cout << "done build, sah cost is " << cuBQL::computeSAH(bvh) << std::endl;
      else
        std::cout << "done build..." << std::endl;

      // CUDAArray<float3> queryPoints;
      // queryPoints.upload(h_queryPoints);
    
      int numQueries = queryPoints.size();
      CUDAArray<float> closest(numQueries);
#if DO_STATS
      CUDAArray<Stats,ManagedMem> stats(1);
      stats.bzero();
#endif

      // ------------------------------------------------------------------
      std::cout << "first query for warm-up and checking reference data (if provided)"
                << std::endl;
      // ------------------------------------------------------------------
      resetResults<<<divRoundUp(numQueries,128),128>>>(closest.data(),numQueries);
      switch(testConfig.knn_k) {
      case 0:
        // no knn, just fcp
        runFCP<<<divRoundUp(numQueries,128),128>>>
          (closest.data(),bvh,data.get(),queryPoints.get(),
           testConfig.maxQueryRadius,numQueries
#if DO_STATS
           ,stats.get()
#endif
           );
        break;
      case 4:
        runKNN<4><<<divRoundUp(numQueries,128),128>>>
          (closest.data(),bvh,data.get(),queryPoints.get(),
           testConfig.maxQueryRadius,numQueries
#if DO_STATS
           ,stats.get()
#endif
           );
        break;
      case 8:
        runKNN<8><<<divRoundUp(numQueries,128),128>>>
          (closest.data(),bvh,data.get(),queryPoints.get(),
           testConfig.maxQueryRadius,numQueries
#if DO_STATS
           ,stats.get()
#endif
           );
        break;
      case 16:
        runKNN<16><<<divRoundUp(numQueries,128),128>>>
          (closest.data(),bvh,data.get(),queryPoints.get(),
           testConfig.maxQueryRadius,numQueries
#if DO_STATS
           ,stats.get()
#endif
           );
        break;
      case 64:
        runKNN<64><<<divRoundUp(numQueries,128),128>>>
          (closest.data(),bvh,data.get(),queryPoints.get(),
           testConfig.maxQueryRadius,numQueries
#if DO_STATS
           ,stats.get()
#endif
           );
        break;
      case 50:
        runKNN<50><<<divRoundUp(numQueries,128),128>>>
          (closest.data(),bvh,data.get(),queryPoints.get(),
           testConfig.maxQueryRadius,numQueries
#if DO_STATS
           ,stats.get()
#endif
           );
        break;
      case 20:
        runKNN<20><<<divRoundUp(numQueries,128),128>>>
          (closest.data(),bvh,data.get(),queryPoints.get(),
           testConfig.maxQueryRadius,numQueries
#if DO_STATS
           ,stats.get()
#endif
           );
        break;
      default:
        throw std::runtime_error("un-supported k="+std::to_string(testConfig.knn_k)+" for knn queries...");
      };
      CUBQL_CUDA_SYNC_CHECK();
#if DO_STATS
      auto results = closest.download();
      CUBQL_CUDA_SYNC_CHECK();
      for (int i=0;i<14;i++) {
        int idx = results.size()-1-(1<<i);
        std::cout << "  result[" << idx << "] = " << results[idx] << std::endl;;
      }
      double sum = 0;
      for (auto r : results)
        sum += r;
      std::cout << "CHECKSUM " << sum << std::endl;
      PRINT(stats.get()->numNodes);
      PRINT(stats.get()->numPrims);
      PRINT(prettyNumber(stats.get()->numNodes));
      PRINT(prettyNumber(stats.get()->numPrims));
      std::cout << "STATS_DIGEST " << (stats.get()->numNodes+stats.get()->numPrims) << std::endl;
      std::cout << "NICE_DIGEST " << prettyNumber(stats.get()->numNodes+stats.get()->numPrims) << std::endl;
      exit(0);
#endif



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
        std::cout << "timing run with " << numPerRun << " repetition(s).." << std::endl;
        double t0 = getCurrentTime();
        for (int i=0;i<numPerRun;i++) {
          resetResults<<<divRoundUp(numQueries,128),128>>>(closest.data(),numQueries);
          runFCP<<<divRoundUp(numQueries,128),128>>>
            (closest.data(),
             bvh,
             data.get(),
             queryPoints.get(),
             testConfig.maxQueryRadius,
             numQueries
#if DO_STATS
             ,stats.get()
#endif
             );
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
    
  } // ::cuBQL::test_rig
} // ::cuBQL

using namespace ::cuBQL::test_rig;

int main(int ac, char **av)
{
  BuildConfig buildConfig;
  std::string bvhType = "binary";
  TestConfig testConfig;
  int numDims = 3;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-dg" || arg == "--data-dist" || arg == "--data-generator") {
      testConfig.dataGen = av[++i];
    } else if (arg == "-dc" || arg == "--data-count") {
      testConfig.dataCount = std::stoi(av[++i]);
    } else if (arg == "-qg" || arg == "--query-generator") {
      testConfig.queryGen = av[++i];
    } else if (arg == "-qc" || arg == "--query-count") {
      testConfig.queryCount = std::stoi(av[++i]);
    } else if (arg == "-nd" || arg == "--num-dims") {
      if (std::string(av[i+1]) == "n") {
        numDims = CUBQL_TEST_N;
        ++i;
      } else
        numDims = std::stoi(av[++i]);
    } else if (arg == "-bt" || arg == "--bvh-type")
      bvhType = av[++i];
    else if (arg == "-sah" || arg == "--sah")
      buildConfig.enableSAH();
    else if (arg == "-elh" || arg == "--elh")
      buildConfig.enableELH();
    else if (arg == "-mr" || arg == "-mqr" || arg == "--max-query-radius")
      testConfig.maxQueryRadius = std::stof(av[++i]);
    else if (arg == "-k" || arg == "--knn-k")
      testConfig.knn_k = std::stoi(av[++i]);
    else if (arg == "--check-reference") {
      testConfig.referenceFileName = av[++i];
      testConfig.make_reference = false;
    } else if (arg == "--dump-test-data") {
      testConfig.dumpTestData = true;
    } else if (arg == "--make-reference") {
      testConfig.referenceFileName = av[++i];
      testConfig.make_reference = true;
      testConfig.maxTimeThreshold = 0.f;
    } else if (arg == "-mlt" || arg == "-lt")
      buildConfig.makeLeafThreshold = std::stoi(av[++i]);
    else
      usage("unknown cmd-line argument '"+arg+"'");
  }

  if (numDims == 2)
    testFCP<2>(testConfig,buildConfig);
  else if (numDims == 3)
    testFCP<3>(testConfig,buildConfig);
  else if (numDims == 4)
    testFCP<4>(testConfig,buildConfig);
#if CUBQL_TEST_N
  else if (numDims == CUBQL_TEST_N)
    testFCP<CUBQL_TEST_N>(testConfig,buildConfig);
#endif
  else
    throw std::runtime_error("unsupported number of dimensions "+std::to_string(numDims));
  return 0;
}
