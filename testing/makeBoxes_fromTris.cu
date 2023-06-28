DEPRECATED

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

  std::cout << "Usage: ./cuBQL_makeBoxes_fromTris tris.bin -o boxes.bin [args]\n\n";
  exit(error.empty()?0:1);
}

int main(int ac, char **av)
{
  std::string outFileName;
  std::string inFileName;
  
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o") {
      outFileName = av[++i];
    } else if (arg[0] == '-')
      usage("unknown cmdline arg '"+arg+"'");
    else
      inFileName = arg;
  }
  if (outFileName.empty())
    usage("no output filename specified");
  if (inFileName.empty())
    usage("no input tris file specified");

  std::cout << "loading tris from '"
            << inFileName << "'" << std::endl;
  std::vector<Triangle> tris = loadData<Triangle>(inFileName);

  std::cout << "making boxes..." << std::endl;
  std::vector<box3f> boxes;
  for(auto tri : tris) {
    box3f bbox;
    bbox.set_empty();
    bbox.grow(tri.a);
    bbox.grow(tri.b);
    bbox.grow(tri.c);
    boxes.push_back(bbox);
  }
  
  std::cout << "saving as " << outFileName << std::endl;
  saveData(boxes,outFileName);
  return 0;
}
