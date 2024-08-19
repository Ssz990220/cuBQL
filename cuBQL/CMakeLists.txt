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

# a interface-only library that only sets include paths etc; when
# using this the user has to manulaly set CUBQL_BUILDER_INSTANTIATION
# in one of his/her source files
add_library(cuBQL_interface INTERFACE)
target_sources(cuBQL_interface INTERFACE
  # main public "interface" to this library
  bvh.h
  # general math struff to make public stuff work
  math/common.h
  math/math.h
  math/vec.h
  math/box.h
  # internal stuff
  builder/cuda/builder_common.h
  builder/cuda/sm_builder.h
  builder/cuda/sah_builder.h
  builder/cuda/gpu_builder.h
  builder/cuda/morton.h
  builder/cuda/rebinMortonBuilder.h
  builder/cuda/wide_gpu_builder.h
  )
target_include_directories(cuBQL_interface INTERFACE
  ${PROJECT_SOURCE_DIR}/
  )

foreach(T IN ITEMS float)
  foreach(D IN ITEMS 2 3 4)
    add_library(cuBQL_cuda_${T}${D} STATIC EXCLUDE_FROM_ALL 
      impl/instantiate_builders.cu
      )
    target_link_libraries(cuBQL_cuda_${T}${D}
      PUBLIC
      cuBQL_interface
      )
    target_compile_definitions(cuBQL_cuda_${T}${D}
      PUBLIC
      -DCUBQL_INSTANTIATE_T=${T}
      -DCUBQL_INSTANTIATE_D=${D}
      )
  endforeach()
endforeach()

add_library(cuBQL_queries INTERFACE)
target_sources(cuBQL_queries INTERFACE
  queries/shrinkingRadiusQuery.h
  queries/fixedBoxQuery.h
  )
target_link_libraries(cuBQL_queries
  INTERFACE
  cuBQL_interface
  )


# builds an actual static library that already contains
# template-instantiations of the builder(s), helper functions, etc
add_library(cuBQL_impl STATIC
  impl/instantiate_builders.cu
  )
target_link_libraries(cuBQL_impl PUBLIC
  cuBQL_interface
  )

