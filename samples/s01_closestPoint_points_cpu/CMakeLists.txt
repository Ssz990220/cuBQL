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

# ==================================================================
# sample that performs gpu-side closest-point queries between sets of
# float3 points
# ==================================================================
add_executable(cuBQL_sample01_points_closestPoint_cpu
  closestPoint.cpp
  )
target_link_libraries(cuBQL_sample01_points_closestPoint_cpu
  # the cpu-side builders for float3 data
  cuBQL_cpu_float3
  # common samples stuff
  cuBQL_samples_common
  )

