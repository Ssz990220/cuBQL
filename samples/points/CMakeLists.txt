# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

foreach(T in float)
  add_executable(cuBQL_pointData_closestPoint_host_${T}
    closestPoint.h
    closestPoint.cpp
    )
  target_link_libraries(cuBQL_pointData_closestPoint_host_${T}
    PUBLIC
    cuBQL_samples_common
    )
  target_compile_definitions(cuBQL_pointData_closestPoint_host_${T}
    PUBLIC
    -DCUBQL_SAMPLES_T=${T}
    )
  
  add_executable(cuBQL_pointData_closestPoint_cuda_${T}
    closestPoint.h
    closestPoint.cu
    )
  target_link_libraries(cuBQL_pointData_closestPoint_cuda_${T}
    PUBLIC
    cuBQL_samples_common
    )
  target_compile_definitions(cuBQL_pointData_closestPoint_host_${T}
    PUBLIC
    -DCUBQL_SAMPLES_T=${T}
    )
  
endforeach()
