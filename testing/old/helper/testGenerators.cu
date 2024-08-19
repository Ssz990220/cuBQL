// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

#include "testing/helper/Generator.h"

using namespace cuBQL::test_rig;

int main(int ac, char **av)
{
  CUDAArray<vec_t<float,3>> points;

  std::cout << "generating 1K of uniform random float3 points" << std::endl;
  UniformPointGenerator<float,3> uni3f;
  uni3f.generate(points,1024,0x12345);
  std::cout << "all good" << std::endl;

  std::cout << "generating 1K of clustered float3 points" << std::endl;
  ClusteredPointGenerator<float,3> cluster3f;
  cluster3f.generate(points,1024,0x12345);
  std::cout << "all good" << std::endl;
  
}
