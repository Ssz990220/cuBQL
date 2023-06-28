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
#include <fstream>

using namespace cuBQL::test_rig;

std::string outFileName = "a.svg";
std::string generatorString = "uniform";
int numBoxes = 10000;

int main(int ac, char **av)
{
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-')
      generatorString = arg;
    else if (arg == "-o")
      outFileName = av[++i];
    else if (arg == "-n" || arg == "-np" || arg == "--num-pints")
      numBoxes = std::stoi(av[++i]);
    else
      throw std::runtime_error("unknown cmd-line arg '"+arg+"'");
  }
  
  typename BoxGenerator<float,2>::SP gen
    = BoxGenerator<float,2>::createFromString(generatorString);
  CUDAArray<box2f> d_boxes = gen->generate(numBoxes,0);
  
  std::vector<box2f> boxes = d_boxes.download();
  box2f bounds;
  for (auto box : boxes)
    bounds.grow(box);
  float sz = max(bounds.size().x,bounds.size().y);
  
  std::ofstream file(outFileName);
  file << "<svg height=\"1000\" width=\"1000\">" << std::endl;
  for (auto box : boxes) {
    vec2f lo = (box.lower - bounds.lower) * (1.f / sz);
    vec2f hi = (box.upper - bounds.lower) * (1.f / sz);
    int x0 = int(1000*lo.x);
    int y0 = int(1000*lo.y);
    int x1 = int(1000*hi.x);
    int y1 = int(1000*hi.y);
    // if (x1 == x0) x1++;
    // if (y1 == y0) y1++;

    file << "<polygon points=\""
         << x0 << "," << y0 << " "
         << x1 << "," << y0 << " "
         << x1 << "," << y1 << " "
         << x0 << "," << y1 << " "
         << x0 << "," << y0 
         << "\" style=\"fill:lime;stroke:black;stroke-width:2\" />"
         << std::endl;
  }
  file << "</svg>" << std::endl;
}
