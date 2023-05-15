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

#include "testing/helper.h"

using namespace testing;

int main(int ac, char **av)
{
  float3 lower = make_float3(0.f,0.f,0.f);
  float3 upper = make_float3(1.f,1.f,1.f);
  std::string outFileName;
  int numPoints = 100000;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o") {
      outFileName = av[++i];
    } else if (arg == "-n") {
      numPoints = std::stoi(av[++i]);
    } else if (arg == "--lower") {
      lower.x = std::stof(av[++i]);
      lower.y = std::stof(av[++i]);
      lower.z = std::stof(av[++i]);
    } else if (arg == "--upper") {
      upper.x = std::stof(av[++i]);
      upper.y = std::stof(av[++i]);
      upper.z = std::stof(av[++i]);
    } else
      throw std::runtime_error("./makePoints_uniform -n numPoints -o outFileName [--lower x y z][--upper x y z]");
  }
  if (outFileName.empty())
    throw std::runtime_error("no output filename specified");

  srand48(computeSeed(outFileName));
  std::vector<float3> points;
  for (int i=0;i<numPoints;i++)
    points.push_back(make_float3(drand48(),drand48(),drand48()));
  saveData(points,outFileName);
  return 0;
}
