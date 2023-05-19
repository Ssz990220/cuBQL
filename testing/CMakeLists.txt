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
  
  add_executable(cuBQL_buildPerf_points buildPerf.cu)
  target_link_libraries(cuBQL_buildPerf_points PUBLIC cuBQL_testing)
  target_compile_definitions(cuBQL_buildPerf_points PUBLIC -DEXPECT_POINTS=1)
    
  add_executable(cuBQL_buildPerf_tris buildPerf.cu)
  target_link_libraries(cuBQL_buildPerf_tris PUBLIC cuBQL_testing)
  target_compile_definitions(cuBQL_buildPerf_tris PUBLIC -DEXPECT_TRIS=1)

  add_executable(cuBQL_makePoints_uniform makePoints_uniform.cu)
  target_link_libraries(cuBQL_makePoints_uniform PUBLIC cuBQL_testing)

  add_executable(cuBQL_makePoints_sampleOBJ makePoints_sampleOBJ.cu)
  target_link_libraries(cuBQL_makePoints_sampleOBJ PUBLIC cuBQL_testing)

  add_executable(cuBQL_makeTris_fromOBJ makeTris_fromOBJ.cu)
  target_link_libraries(cuBQL_makeTris_fromOBJ PUBLIC cuBQL_testing)

endif()
