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

#include "samples/common/Generator.h"
#include "samples/common/CmdLine.h"

std::string generator   = "uniform";
int         numPoints   = 100000;
std::string outFileName = "cuBQL.dat";
int         dataDim     = 3;
int         seed        = 290374;

void usage(const std::string &error)
{
  if (!error.empty())
    std::cerr << "Error: " << error << "\n\n";
  std::cout << "Usage: ./cuBQL_generatePoints [args]*\n";
  std::cout << "w/ args:\n";
  std::cout << " -n int:numPoints\n";
  std::cout << " -d dims(1,2,3,4,n)\n"; 
  std::cout << " -g generator        ; see README for generator strings\n";
  std::cout << " -o outFileName\n";
  
  exit(error.empty() ? 0 : 1);
}


template<int D>
void run()
{
  using namespace cuBQL;
  using namespace cuBQL::samples;
  
  std::cout << "#cuBQL.genPoints: creating generator '" << ::generator << "'" << std::endl;
  typename PointGenerator<D>::SP generator
    = PointGenerator<D>::createFromString(::generator);
  std::cout << "#cuBQL.genPoints: generating '" << numPoints
            << " points w/ seed " << seed << std::endl;
  std::vector<vec_t<double,D>> points
    = generator->generate(numPoints,seed);
  std::cout << "#cuBQL.genPoints: saving to " << outFileName << std::endl;
  saveBinary(outFileName,points);
  std::cout << "#cuBQL.genPoints: all done." << std::endl;
}

int main(int ac, char **av)
{
  cuBQL::samples::CmdLine cmdLine(ac,av);
  while (!cmdLine.consumed()) {
    const std::string arg = cmdLine.getString();
    if (arg == "-d" || arg == "--dim") {
      dataDim = cmdLine.getInt();
    } else if (arg == "-n" || arg == "--num") {
      numPoints = cmdLine.getInt();
    } else if (arg == "-s" || arg == "--seed") {
      seed = cmdLine.getInt();
    } else if (arg == "-o" || arg == "--out") {
      outFileName = cmdLine.getString();
    } else if (arg == "-g" || arg == "--generator") {
      generator = cmdLine.getString();
    } else
      usage("unknown cmd-line argument '"+arg+"'");
  }
  if (dataDim == 2)
    run<2>();
  else if (dataDim == 3)
    run<3>();
  else if (dataDim == 4)
    run<4>();
#if CUBQL_USER_DIM
  else if (dataDim == CUBQL_USER_DIM || dataDim == -1)
    run<CUBQL_USER_DIM>();
#endif
  else
    usage("un-supported data dimensionality '"+std::to_string(dataDim)+"'");
  return 0;
}

