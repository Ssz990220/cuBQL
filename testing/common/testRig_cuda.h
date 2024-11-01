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

#include "testing/common/testRig_host.h"

namespace cuBQL {
  namespace testRig {

    // ******************************************************************
    // INTERFACE
    // (which functionality this header file provides)
    // ******************************************************************

    /*! device abstraction that implements a 'cuda device'; i.e., one
        where all memory allocations, uploads, downloads, etc, target
        GPUd evice memory of a CUDA-enabled GPU */
    struct CUDADevice : public DeviceAbstraction {
      void free(const void *ptr) override;
      void *malloc(size_t numBytes) override;
      void upload(void *d_mem, void *h_mem, size_t numBytes) override;
      void download(void *h_mem, void *d_mem, size_t numBytes) override;
    };

    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************
    
    void CUDADevice::free(const void *ptr) 
    {
      CUBQL_CUDA_CALL(Free((void *)ptr));
    }
      
    void *CUDADevice::malloc(size_t numBytes) 
    {
      void *ptr = 0;
      CUBQL_CUDA_CALL(Malloc(&ptr,numBytes));
      return ptr;
    }
    
    void CUDADevice::upload(void *d_mem, void *h_mem, size_t numBytes) 
    {
      CUBQL_CUDA_CALL(Memcpy(d_mem,h_mem,numBytes,cudaMemcpyDefault)); 
    }
    
    void CUDADevice::download(void *h_mem, void *d_mem, size_t numBytes) 
    {
      CUBQL_CUDA_CALL(Memcpy(h_mem,d_mem,numBytes,cudaMemcpyDefault)); 
    }
    
  }
}

