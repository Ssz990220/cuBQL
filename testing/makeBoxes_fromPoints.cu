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

#include "testing/helper/triangles.h"

using namespace testing;

void usage(const std::string &error = "")
{
  if (!error.empty())
    std::cout << "Error: " << error << "\n\n";

  std::cout << "Usage: ./cuBQL_makeBoxes_fromPoints points.bin -o boxes.bin [args]\n\n";
  std::cout << "w/ args:\n";
  std::cout << " -o outFileName\n";
  std::cout << " --obj         : save as a obj file, not a data file\n";
  std::cout << " -bs box-size  : width of each generated box\n";
  exit(error.empty()?0:1);
}

int main(int ac, char **av)
{
  std::string outFileName;
  std::string inFileName;
  
  float boxSize = 0.f;
  bool useOBJ = false;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o") {
      outFileName = av[++i];
    } else if (arg == "--obj" || arg == "-obj") {
      useOBJ = true;
    } else if (arg == "--box-size" || arg == "-bs") {
      boxSize = std::stof(av[++i]);
    } else if (arg[0] == '-')
      usage("unknown cmdline arg '"+arg+"'");
    else
      inFileName = arg;
  }
  if (outFileName.empty())
    usage("no output filename specified");
  if (inFileName.empty())
    usage("no input points file specified");

  if (outFileName.substr(outFileName.size()-4) == ".obj")
    useOBJ = true;
                         
  std::cout << "loading points from '"
            << inFileName << "'" << std::endl;
  std::vector<float3> points = loadData<float3>(inFileName);

  if (boxSize <= 0.f) {
    std::cout << "no box size specified; computing one automatically ..." << std::endl;
    box3f bbox;
    bbox.set_empty();
    for (auto p : points) bbox.grow(p);
    boxSize = length(bbox.upper-bbox.lower);
    boxSize /= sqrtf(points.size());
  }

  std::cout << "making boxes..." << std::endl;
  std::vector<box3f> boxes;
  for(auto p : points) 
    boxes.push_back(make_box3f(p-make_float3(.5f*boxSize),
                               p+make_float3(.5f*boxSize)));

  
  if (useOBJ) {
    std::cout << "saving as OBJ, to " << outFileName << std::endl;
    saveOBJ(triangulate(boxes),outFileName);
  } else {
    std::cout << "saving as " << outFileName << std::endl;
    saveData(boxes,outFileName);
  }  
  return 0;
}
