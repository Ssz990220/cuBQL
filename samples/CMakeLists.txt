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

# common stuff - data generators and IO for loading/generating test data
add_subdirectory(common)

# samples of various queries that operate on *point* data sets
add_subdirectory(points)







#add_executable(cuBQL_lineSegs3f lineSegs3f.cu)
#target_link_libraries(cuBQL_lineSegs3f cuBQL_interface)

#add_executable(cuBQL_fcpTriangles fcpTriangles.cu)
#target_link_libraries(cuBQL_fcpTriangles cuBQL_testing cuBQL_interface)


