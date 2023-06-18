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

add_library(cuBQL-unit-tests INTERFACE)
target_compile_definitions(cuBQL-unit-tests INTERFACE -DCUBQL_TEST_N=${CUBQL_TEST_N})
target_link_libraries(cuBQL-unit-tests INTERFACE cuBQL_interface)



add_executable(instantiate-vec_t instantiate-vec_t.cu)
target_link_libraries(instantiate-vec_t cuBQL-unit-tests)

add_executable(instantiate-box_t instantiate-box_t.cu)
target_link_libraries(instantiate-box_t cuBQL-unit-tests)

add_executable(instantiate-bvh_t instantiate-bvh_t.cu)
target_link_libraries(instantiate-bvh_t cuBQL-unit-tests)

foreach(N IN ITEMS 2 3 4)
  add_executable(instantiate-binaryBVH-builders-${N} instantiate-binaryBVH-builders.cu)
  target_link_libraries(instantiate-binaryBVH-builders-${N} cuBQL-unit-tests)
  target_compile_definitions(instantiate-binaryBVH-builders-${N} PUBLIC -DUNIT_TEST_N_FROM_CMAKE=${N})
endforeach()


  
