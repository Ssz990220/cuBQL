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

#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

namespace cuBQL {
  namespace samples {

    /*! load a vector of (binary) data from a binary-dump file. the
        data file is supposed to start with a size_t that specifies
        the *number* N of elements to be expected in that file,
        followed by the N "raw"-binary data items */
    template<typename T>
    std::vector<T> loadBinary(const std::string &fileName)
    {
      std::ifstream in(fileName.c_str(),std::ios::binary);
      if (!in.good())
        throw std::runtime_error("could not open '"+fileName+"'");
      size_t count;
      in.read((char*)&count,sizeof(count));

      std::vector<T> data(count);
      in.read((char*)data.data(),count*sizeof(T));
      return data;
    }

    /*! write a vector of (binary) data into a binary-dump file. the
        data file is supposed to start with a size_t that specifies
        the *number* N of elements to be expected in that file,
        followed by the N "raw"-binary data items */
    template<typename T>
    void saveBinary(const std::string &fileName,
                    const std::vector<T> &data)
    {
      std::ofstream out(fileName.c_str(),std::ios::binary);
      size_t count = data.size();
      out.write((char*)&count,sizeof(count));
      
      out.write((char*)data.data(),count*sizeof(T));
    }

  }
}
