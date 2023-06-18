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

#include "testing/helper/Generator.h"
#include <random>

namespace testing {

  /*! literal re-implementation of the stdlib 'drand48()' LCG
    generator. note this is usually significantly worse than the
    owl::common::LCG class above */
  struct DRand48
  {
    /*! initialize the random number generator with a new seed (usually
      per pixel) */
    inline __both__ void init(int seed = 0)
    {
      state = seed;
      for (int warmUp=0;warmUp<10;warmUp++)
        (*this)();
    }

    /*! get the next 'random' number in the sequence */
    inline __both__ float operator() ()
    {
      const uint64_t a = 0x5DEECE66DULL;
      const uint64_t c = 0xBULL;
      const uint64_t mask = 0xFFFFFFFFFFFFULL;
      state = a*state + c;
      return float(state & mask) / float(mask+1ULL);
      //return ldexpf(float(state & mask), -24);
    }

    uint64_t state;
  };

  struct IRand48
  {
    /*! initialize the random number generator with a new seed (usually
      per pixel) */
    inline __both__ void init(int seed = 0)
    {
      state = seed;
      for (int warmUp=0;warmUp<10;warmUp++)
        (*this)();
    }

    /*! get the next 'random' number in the sequence */
    inline __both__ uint32_t operator() ()
    {
      const uint64_t a = 0x5DEECE66DULL;
      const uint64_t c = 0xBULL;
      // const uint64_t mask = 0xFFFFFFFFFFFFULL;
      state = a*state + c;
      // return float(state & mask) / float(mask+1ULL);
      //return ldexpf(float(state & mask), -24);
      return uint32_t(state);
    }

    uint64_t state;
  };
  

  // ==================================================================
  // uniform points
  // ==================================================================

  template<typename T> inline __both__ T uniform_default_lower();
  template<typename T> inline __both__ T uniform_default_upper();
  template<> inline __both__ float uniform_default_lower<float>() { return 0.f; }
  template<> inline __both__ float uniform_default_upper<float>() { return 1.f; }
  
  template<typename T, int D>
  __global__
  void uniformPointGenerator(vec_t<T,D> *d_points, int count, int seed)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= count) return;
    vec_t<T,D> &mine = d_points[tid];

    DRand48 rng;
    rng.init(seed+tid);

    T lo = uniform_default_lower<T>();
    T hi = uniform_default_upper<T>();
    
    for (int i=0;i<D;i++)
      mine[i] = T(lo + rng() * (hi-lo));
  }

  template<typename T, int D>
  CUDAArray<vec_t<T,D>> UniformPointGenerator<T,D>::generate(int count, int seed)
  {
    CUDAArray<vec_t<T,D>> res;
    res.resize(count);
    int bs = 1024;
    int nb = divRoundUp(int(count),bs);
    uniformPointGenerator<T,D><<<nb,bs>>>(res.data(),count,seed);
    return res;
  }

  template struct UniformPointGenerator<float,2>;
  template struct UniformPointGenerator<float,3>;
  template struct UniformPointGenerator<float,4>;
#if CUBQL_TEST_N
  template struct UniformPointGenerator<float,CUBQL_TEST_N>;
#endif



  
  // ==================================================================
  // clustered points
  // ==================================================================


  // template<typename T, int D>
  // __global__
  // void clusteredPointGenerator(vec_t<T,D> *d_points, int count, int seed)
  // {
  //   int tid = threadIdx.x+blockIdx.x*blockDim.x;
  //   if (tid >= count) return;
  //   vec_t<T,D> &mine = d_points[tid];

  //   IRand48 irand;
  //   irand.init(seed+tid);

  //   int numClusters = int(1+sqrtf(numPoints));
                                 
  //   int clusterID = irand() % numClusters;
  //   DRand48 frand;
  //   frand.init(seed+0x123+clusterID);

  //   vec_t<float,D> clusterCenter;
  //   for (int i=0;i<D;i++)
  //     clusterCenter[i] = frand();

  //   float clusterWidth = 1.f/numClusters;

  //   frand.init(tid+seed);
  //   float u,v,w;
  //   while (1) {
  //     u = 2.f*frand()-1.f;
  //     v = 2.f*frand()-1.f;
  //     w = u*u+v*v;
  //     if (w <= 1.f) break;
  //   }
  //   float z = sqrtf(-2.f*logf(w)/w);

    

  //   T lo = clustered_default_lower<T>();
  //   T hi = clustered_default_upper<T>();
    
  //   for (int i=0;i<D;i++)
  //     mine[i] = T(lo + rng() * (hi-lo));
  // }

  template<typename T, int D>
  CUDAArray<vec_t<T,D>> ClusteredPointGenerator<T,D>::generate(int count, int seed)
  {
    std::default_random_engine rng;
    rng.seed(seed);
    std::uniform_real_distribution<double> uniform(0.f,1.f);
  
    int numClusters = int(1+sqrtf(count));
    std::vector<vec_t<float,D>> clusterCenters;
    for (int cc=0;cc<numClusters;cc++) {
      vec_t<float,D> c;
      for (int i=0;i<D;i++)
        c[i] = uniform(rng);
      clusterCenters.push_back(c);
    }
    
    std::normal_distribution<double> gaussian(0.f,1.f/numClusters);
    std::vector<vec_t<T,D>> points;
    for (int sID=0;sID<count;sID++) {
      int clusterID = int(uniform(rng)*numClusters) % numClusters;
      vec_t<float,D> pt;
      for (int i=0;i<D;i++)
        pt[i] = gaussian(rng) + clusterCenters[clusterID][i];
      points.push_back(pt);
    }
    
    CUDAArray<vec_t<T,D>> res;
    res.upload(points);
    return res;
  }

  template struct ClusteredPointGenerator<float,2>;
  template struct ClusteredPointGenerator<float,3>;
  template struct ClusteredPointGenerator<float,4>;
#if CUBQL_TEST_N
  template struct ClusteredPointGenerator<float,CUBQL_TEST_N>;
#endif
  // ==================================================================
}
