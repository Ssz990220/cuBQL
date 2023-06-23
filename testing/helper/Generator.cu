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
#include <exception>

namespace testing {

  namespace tokenizer {
    std::string findFirst(const char *curr, const char *&endOfFound)
    {
      if (curr == 0)
        return "";
      
      while (*curr && strchr(" \t\r\n",*curr)) {
        ++curr;
      }
      if (*curr == 0) {
        endOfFound = curr;
        return "";
      }
      
      std::stringstream ss;
      if (strchr("{}():,",*curr)) {
        ss << *curr++;
      } else if (isalnum(*curr)) {
        while (isalnum(*curr)) {
          ss << *curr;
          curr++;
        }
      }
      else
        throw std::runtime_error("unable to parse ... '"+std::string(curr)+"'");
      
      endOfFound = curr;
      return ss.str();
    }
  };
  
  template<typename T> inline T to_scalar(const std::string &s);
  template<> inline float to_scalar<float>(const std::string &s)
  { return std::stof(s); }
  template<> inline int to_scalar<int>(const std::string &s)
  { return std::stoi(s); }

  // ==================================================================
  // point generator base
  // ==================================================================
  template<typename T, int D>
  typename PointGenerator<T,D>::SP
  PointGenerator<T,D>::createAndParse(const char *&curr)
  {
    const char *next = 0;
    PING; PRINT(curr);
    std::string type = tokenizer::findFirst(curr,next);
    PRINT(type);
    PRINT(next);
    if (type == "") throw std::runtime_error("could not parse generator type");

    typename PointGenerator<T,D>::SP gen;
    if (type == "uniform")
      gen = std::make_shared<UniformPointGenerator<T,D>>();
    else if (type == "clustered")
      gen = std::make_shared<ClusteredPointGenerator<T,D>>();
    else if (type == "nrooks")
      gen = std::make_shared<NRooksPointGenerator<T,D>>();
    else if (type == "clustered")
      gen = std::make_shared<RemapPointGenerator<T,D>>();
    else if (type == "remap")
      gen = std::make_shared<RemapPointGenerator<T,D>>();
    else
      throw std::runtime_error("un-recognized point generator type '"+type+"'");
    curr = next;
    PING;
    PRINT(curr);
    PRINT(next);
    gen->parse(curr);
    PING;
    PRINT(curr);
    PRINT(next);
    return gen;
  }


  template<typename T, int D>
  typename PointGenerator<T,D>::SP
  PointGenerator<T,D>::createFromString(const std::string &wholeString)
  {
    PING; PRINT(wholeString);
    const char *curr = wholeString.c_str(), *next = 0;
    SP generator = createAndParse(curr);
    PING; PRINT(curr); PRINT((int*)next);
    PRINT("'"+std::string(curr)+"'");
    std::string trailing = tokenizer::findFirst(curr,next);
    PRINT(trailing);
    if (!trailing.empty())
      throw std::runtime_error("un-recognized trailing stuff '"
                               +std::string(curr)
                               +"' at end of point generator string");
    return generator;
  }
  
  template<typename T, int D>
  void PointGenerator<T,D>::parse(const char *&currentParsePos)
  {}

  template struct PointGenerator<float,2>;
  template struct PointGenerator<float,3>;
  template struct PointGenerator<float,4>;
#if CUBQL_TEST_N
  template struct PointGenerator<float,CUBQL_TEST_N>;
#endif
  // ==================================================================

  
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
    if (count <= 0)
      throw std::runtime_error("UniformPointGenerator<T,D>::generate(): invalid count...");
    // PING; PRINT(count);
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
  // re-mapping
  // ==================================================================
  
  template<typename T, int D>
  __global__
  void remapPointsKernel(vec_t<T,D> *points,
                         int count,
                         vec_t<T,D> lower,
                         vec_t<T,D> upper)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= count) return;

    auto &point = points[tid];
    for (int d=0;d<D;d++)
      point[d]
        = lower[d]
        + T(point[d]
            * (typename dot_result_t<T>::type)(upper[d]-lower[d])
            / (uniform_default_upper<T>() - uniform_default_lower<T>()));
  }
  
  template<typename T, int D>
  RemapPointGenerator<T,D>::RemapPointGenerator()
  {
    for (int d=0;d<D;d++) {
      lower[d] = uniform_default_lower<T>();
      upper[d] = uniform_default_upper<T>();
    }
  }

  template<typename T, int D>
  CUDAArray<vec_t<T,D>> RemapPointGenerator<T,D>::generate(int count, int seed)
  {
    if (!source)
      throw std::runtime_error("RemapPointGenerator: no source defined");
    CUDAArray<vec_t<T,D>> pts = source->generate(count,seed);

    int bs = 128;
    int nb = divRoundUp(count,bs);
    remapPointsKernel<<<nb,bs>>>(pts.get(),count,lower,upper);
    
    return pts;
  }

  template<typename T, int D>
  void RemapPointGenerator<T,D>::parse(const char *&currentParsePos)
  {
    const char *next = 0;
    for (int d=0;d<D;d++) {
      std::string tok = tokenizer::findFirst(currentParsePos,next);
      assert(tok != "");
      lower[d] = to_scalar<T>(tok);
      currentParsePos = next;
    }
    for (int d=0;d<D;d++) {
      std::string tok = tokenizer::findFirst(currentParsePos,next);
      assert(tok != "");
      upper[d] = to_scalar<T>(tok);
      currentParsePos = next;
    }
    source = PointGenerator<T,D>::createAndParse(currentParsePos);
 }

  
  
  // ==================================================================
  // clustered points
  // ==================================================================

  template<typename T, int D>
  CUDAArray<vec_t<T,D>> ClusteredPointGenerator<T,D>::generate(int count, int seed)
  {
    std::default_random_engine rng;
    rng.seed(seed);
    std::uniform_real_distribution<double> uniform(0.f,1.f);
  
    int numClusters
      = int(1+count/50.f);
      // = int(1+sqrtf(count));
      // = this->numClusters
      // ? this->numClusters
      // : int(1+sqrtf(count));
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
  
  template<typename T, int D>
  CUDAArray<vec_t<T,D>> NRooksPointGenerator<T,D>::generate(int count, int seed)
  {
    int numClusters = (int)(1+count/(float)50);
    // T lo = uniform_default_lower<T>();
    // T hi = uniform_default_upper<T>();
    
    // for (int i=0;i<D;i++)
    //   mine[i] = T(lo + rng() * (hi-lo));
    IRand48 irand;
    irand.init(0x1234321+seed+1);
    DRand48 frand;
    irand.init(0x1234321+seed+2);
    std::vector<vec_t<float,D>> clusterLower(numClusters);
    for (int d=0;d<D;d++) {
      for (int i=0;i<numClusters;i++) {
        clusterLower[i][d] = i/(float)numClusters;
      }
      for (int i=numClusters-1;i>0;--i) {
        int o = irand() % i;
        std::swap(clusterLower[i][d],clusterLower[o][d]);
      }
    }
    
    std::vector<vec_t<float,D>> points(count);
    for (int i=0;i<count;i++) {
      int clusterID = irand() % numClusters;
      for (int d=0;d<D;d++)
        points[i][d] = clusterLower[clusterID][d] + (1.f/numClusters)*frand();
    }
    
    CUDAArray<vec_t<T,D>> res(count);
    // int bs = 128;
    // int nb = divRoundUp(count,bs);
    // nRooksKernel<<<nb,bs>>>(res.data(),count,seed,numClusters);
    // CUDA_SYNC_CHECK();
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
