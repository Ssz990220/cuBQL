# ======================================================================== #
# Copyright 2023-2023 Ingo Wald                                            #
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

add_executable(cuBQL_fcp fcp.cu)
target_link_libraries(cuBQL_fcp PUBLIC cuBQL)

if (NOT CUBQL_IS_SUBMODULE)
  add_library(cuBQL_testing 
    helper/triangles.cu
    )
  target_link_libraries(cuBQL_testing PUBLIC cuBQL)
  
  add_executable(makePoints_uniform makePoints_uniform.cu)
  target_link_libraries(makePoints_uniform PUBLIC cuBQL_testing cuBQL)

  add_executable(makePoints_sampleOBJ makePoints_sampleOBJ.cu)
  target_link_libraries(makePoints_sampleOBJ PUBLIC cuBQL_testing)
endif()
