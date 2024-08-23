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

#include <cstring>

namespace cuBQL {
  namespace testRig {

    // ******************************************************************
    // INTERFACE
    // (which functionality this header file provides)
    // ******************************************************************

    /*! abstraction for a "device" that can allocate memory, and
        upload/download data to/from that device. we use this to
        'virtualize' whether a given sample runs on a GPU or on the
        host */
    struct DeviceAbstraction {
      virtual void *malloc(size_t numBytes) = 0;
      virtual void upload(void *d_mem, void *h_mem, size_t numBytes) = 0;
      virtual void download(void *h_mem, void *d_mem, size_t numBytes) = 0;
      virtual void free(const void *ptr) = 0;

      template<typename T>
      T *alloc(size_t numElems);
      
      template<typename T>
      std::vector<T> download(const T *d_data, size_t numData);
      
      template<typename T>
      T *upload(const std::vector<T> &vec);
      
    };

    /*! device abstration for the host itself. though in that case the
        builders, kernels, and samples could actually directly operate
        on the data (which is already in host memory), for the sake of
        simplicity and readability we rather create a device
        abstraction in which host memory appears in exactly the same
        way as device memory. */
    struct HostDevice : public DeviceAbstraction {
      void *malloc(size_t numBytes) override;
      void upload(void *d_mem, void *h_mem, size_t numBytes) override;
      void download(void *h_mem, void *d_mem, size_t numBytes) override;
      void free(const void *ptr) override;
    };
    
    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************
    
    inline void *HostDevice::malloc(size_t numBytes)
    {
      return ::malloc(numBytes);
    }
    
    inline void HostDevice::upload(void *d_mem, void *h_mem, size_t numBytes)
    {
      ::memcpy(d_mem,h_mem,numBytes);
    }
      
    inline void HostDevice::download(void *h_mem, void *d_mem, size_t numBytes)
    {
      ::memcpy(h_mem,d_mem,numBytes);
    }
      
    inline void HostDevice::free(const void *ptr)
    { ::free((void *)ptr); }

    
    template<typename T>
    T *DeviceAbstraction::alloc(size_t numElems)
    {
      return (T*)malloc(numElems*sizeof(T));
    }
      
    template<typename T>
    T *DeviceAbstraction::upload(const std::vector<T> &vec)
    {
      T *ptr = alloc<T>(vec.size());
      this->upload((void*)ptr,(void*)vec.data(),vec.size()*sizeof(T));
      return ptr;
    }

    template<typename T>
    std::vector<T> DeviceAbstraction::download(const T *d_data, size_t numData)
    {
      std::vector<T> vec(numData);
      this->download((void*)vec.data(),(void*)d_data,vec.size()*sizeof(T));
      return vec;
    }

  }
}

