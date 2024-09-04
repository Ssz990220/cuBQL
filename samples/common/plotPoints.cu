// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
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

#include "cuBQL/math/box.h"
#include "Generator.h"
#include "IO.h"
#include <fstream>

using namespace cuBQL;
using namespace cuBQL::samples;

std::string outFileName = "a.svg";
std::string generatorString = "uniform";
int dataCount = 10000;

int main(int ac, char **av)
{
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-')
      generatorString = arg;
    else if (arg == "-o")
      outFileName = av[++i];
    else if (arg == "-dc" || arg == "--data-count")
      dataCount = std::stoi(av[++i]);
    else if (arg == "-dg" || arg == "--data-generator")
      generatorString = arg;
    else
      throw std::runtime_error("unknown cmd-line arg '"+arg+"'");
  }
  
  typename PointGenerator<2>::SP gen
    = PointGenerator<2>::createFromString(generatorString);
  std::vector<vec2f> points = convert<float,2>(gen->generate(dataCount,290374));
  
  // std::vector<vec2d> points = d_points.download();
  box2f bounds;
  for (auto pt : points)
    bounds.grow(vec2f(pt));
  float sz = max(bounds.size().x,bounds.size().y);
  for (auto &pt : points) {
    pt = (pt - bounds.lower) * (1.f / sz);
    pt.y = 1.f - pt.y;
  }
#if 1
  
  std::ofstream file(outFileName);
  file << "<svg height=\"1000\" width=\"1000\">" << std::endl;
  for (auto pt : points) {
    int x = int(1000*pt.x);
    int y = int(1000*pt.y);
    file << " <circle cx=\"" << x << "\" cy=\"" << y << "\" r=\"2\" stroke=\"black\" stroke-width=\"3\" fill=\"red\" />" << std::endl;
  }
  file << "</svg>" << std::endl;
#else
  std::ofstream file(outFileName);
  file << "#FIG 3.2  Produced by xfig version 3.2.7b" << std::endl;
  file << "Landscape" << std::endl;
  file << "Center" << std::endl;
  file << "Metric" << std::endl;
  file << "A4" << std::endl;
  file << "100.00" << std::endl;
  file << "Single" << std::endl;
  file << "-2" << std::endl;
  file << "1200 2" << std::endl;

  float scale = 10000.f;
  int thick = 3;
  for (auto pt : points) {
    int x0 = int(scale * pt.x);
    int y0 = int(scale * pt.y);
    int x1 = x0+thick;
    int y1 = y0+thick;
    file << "2 2 0 " << thick << " 0 7 50 -1 -1 0.000 0 0 -1 0 0 5" << std::endl;
    file << "\t";
    file << " " << x0 << " " << y0;
    file << " " << x1 << " " << y0;
    file << " " << x1 << " " << y1;
    file << " " << x0 << " " << y1;
    file << " " << x0 << " " << y0;
    file << std::endl;
  }
#endif
}
