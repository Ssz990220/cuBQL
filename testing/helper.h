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
#include <fstream>
#include <vector>

namespace cuBQL {
  namespace test_rig {

    using cuBQL::divRoundUp;
    using cuBQL::box3f;
    using cuBQL::getCurrentTime;
    using cuBQL::prettyNumber;
    using cuBQL::prettyDouble;
    using namespace cuBQL;
  
    /*! computes a seed value for a random number generator, in a way
      that is reproducible for same filename, but different for
      different file names */
    inline size_t computeSeed(const std::string &fileName)
    {
      size_t seed = 0x1234565;
      for (int i=0;i<fileName.size();i++)
        seed = 13 * seed + fileName[i];
      return seed;
    }

    template<typename>
    struct is_std_vector : std::false_type {};
  
    template<typename T, typename A>
    struct is_std_vector<std::vector<T,A>> : std::true_type {};

    template<typename T>
    T read(std::istream &in);
  
    template<typename T>
    void read(std::istream &in, std::vector<T> &vt)
    {
      size_t count;
      in.read((char *)&count,sizeof(count));
      vt.resize(count);
      for (auto &tt : vt) tt = read<T>(in);
    }

    template<typename T>
    void read(std::istream &in, T &t)
    {
      in.read((char *)&t,sizeof(t));
    }
  
    template<typename T>
    T read(std::istream &in)
    {
      T t;
      read(in,t);
      return t;
    }
  
    template<typename T>
    T read_from(const std::string &fn)
    {
      std::ifstream in(fn,std::ios::binary);
      T t;
      read(in,t);
      return t;
    }
  
    template<typename T>
    void write(std::ostream &out, const T &data)
    {
      out.write((const char *)&data,sizeof(data));
    }
    template<typename T>
    void write(std::ostream &out, const std::vector<T> &data)
    {
      size_t count = data.size();;
      out.write((char*)&count,sizeof(count));
      for (auto &v : data)
        write(out,v);
    }

    template<typename T>
    std::vector<T> loadData(const std::string &fileName)
    {
      std::ifstream in(fileName.c_str(),std::ios::binary);
      size_t count;
      in.read((char*)&count,sizeof(count));

      std::vector<T> data(count);
      in.read((char*)data.data(),count*sizeof(T));
      return data;
    }

    template<typename T>
    void saveData(const std::vector<T> &data, const std::string &fileName)
    {
      std::ofstream out(fileName.c_str(),std::ios::binary);
      size_t count = data.size();
      out.write((char*)&count,sizeof(count));

      out.write((const char*)data.data(),count*sizeof(T));
    }


  } // ::cuBQL::tests
} // ::cuBQL
