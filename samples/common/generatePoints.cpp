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
std::string dataType    = "float";
int         dataDim     = 3;
int         seed        = 290374;

void usage(const std::string &error)
{
  if (!error.empty())
    std::cerr << "Error: " << error << "\n\n";
  std::cout << "Usage: ./cuBQL_generatePoints [args]*\n";
  std::cout << "w/ args:\n";
  std::cout << " -n int:numPoints\n";
  std::cout << " -t type(float|int)\n";
  std::cout << " -d dims(1,2,3,4,n)\n";
  std::cout << " -g generator        ; see README for generator strings\n";
  std::cout << " -o outFileName\n";
  
  exit(error.empty() ? 0 : 1);
}


template<typename T, int D>
void run()
{
  using namespace cuBQL;
  using namespace cuBQL::samples;
  
  std::cout << "#cuBQL.genPoints: creating generator '" << ::generator << std::endl;
  typename PointGenerator<T,D>::SP generator
    = PointGenerator<T,D>::createFromString(::generator);
  std::cout << "#cuBQL.genPoints: generating '" << numPoints
            << " points w/ seed " << seed << std::endl;
  std::vector<vec_t<T,D>> points
    = generator->generate(numPoints,seed);
  std::cout << "#cuBQL.genPoints: saving to " << outFileName << std::endl;
  saveBinary(outFileName,points);
  std::cout << "#cuBQL.genPoints: all done." << std::endl;
}

template<typename T>
void run_t()
{
  if (dataDim == 2)
    run<T,2>();
  else if (dataDim == 3)
    run<T,3>();
  else if (dataDim == 4)
    run<T,4>();
#if CUBQL_USER_DIM
  else if (dataDim == CUBQL_USER_DIM || dataDim == -1)
    run<T,CUBQL_USER_DIM>();
#endif
  else
    usage("un-supported data dimensionality '"+std::to_string(dataDim)+"'");
}

int main(int ac, char **av)
{
  cuBQL::samples::CmdLine cmdLine(ac,av);
  while (!cmdLine.consumed()) {
    const std::string arg = cmdLine.getString();
    if (arg == "-t" || arg == "--type") {
      dataType = cmdLine.getString();
    } else if (arg == "-d" || arg == "--dim") {
      dataDim = cmdLine.getInt();
    } else if (arg == "-s" || arg == "--seed") {
      seed = cmdLine.getInt();
    } else if (arg == "-o" || arg == "--out") {
      outFileName = cmdLine.getString();
    } else if (arg == "-g" || arg == "--generator") {
      generator = cmdLine.getString();
    } else
      usage("unknown cmd-line argument '"+arg+"'");
  }
  if (dataType == "float" || dataType == "f")
    run_t<float>();
  else if (dataType == "int" || dataType == "i")
    run_t<int>();
  else if (dataType == "double" || dataType == "d")
    run_t<int>();
  else if (dataType == "long" || dataType == "l")
    run_t<int>();
  else
    usage("unknown or unsupported data type '"+dataType+"'");
  return 0;
}

