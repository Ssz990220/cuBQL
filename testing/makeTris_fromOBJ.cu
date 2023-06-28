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

int main(int ac, char **av)
{
  std::string outFileName;
  std::string inFileName;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o") {
      outFileName = av[++i];
    } else if (arg[0] == '-')
      throw std::runtime_error("./cuBQL_makeTris_fromOBJ inFile.obj -o outFileName");
    else
      inFileName = arg;
  }
  if (outFileName.empty())
    throw std::runtime_error("no output filename specified");
  if (inFileName.empty())
    throw std::runtime_error("no input (obj) filename specified");

  srand48(computeSeed(outFileName));
  
  std::cout << "attempting to load triangles from '"
            << inFileName << "'" << std::endl;
  std::vector<Triangle> triangles = loadOBJ(inFileName);
  std::cout << "done loading '" << inFileName
            << "', got " << prettyNumber(triangles.size())
            << std::endl;

  std::cout << "saving " << triangles.size()
            << " triangles to " << outFileName << std::endl;
  saveData(triangles,outFileName);
  
  return 0;
}
