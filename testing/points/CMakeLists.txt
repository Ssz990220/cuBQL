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

foreach(T IN ITEMS float int)
  foreach(D IN ITEMS 2 3 4)
    # add_executable(cuBQL_pointData_closestPoint_host_${T}${D}
    #   closestPoint.h
    #   closestPoint.cpp
    #   )
    # target_link_libraries(cuBQL_pointData_closestPoint_host_${T}${D}
    #   PUBLIC
    #   cuBQL_testing_common
    #   )
    # target_compile_definitions(cuBQL_pointData_closestPoint_host_${T}${D}
    #   PUBLIC
    #   -DCUBQL_TEST_T=${T}
    #   -DCUBQL_TEST_D=${D}
    #   )
    
    add_executable(cuBQL_pointData_closestPoint_cuda_${T}${D}
      closestPoint.h
      closestPoint.cu
      )
    target_link_libraries(cuBQL_pointData_closestPoint_cuda_${T}${D}
      PUBLIC
      cuBQL_testing_common
      cuBQL_cuda_${T}${D}
      )
    target_compile_definitions(cuBQL_pointData_closestPoint_cuda_${T}${D}
      PUBLIC
      -DCUBQL_TEST_T=${T}
      -DCUBQL_TEST_D=${D}
      )
  endforeach()
endforeach()
