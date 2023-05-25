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
#include <random>

using namespace testing;

int main(int ac, const char **av)
{
  std::string outFileName, queryFileName;
  std::vector<std::string> inFileNames;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-') {
      inFileNames.push_back(arg);
    } else if (arg == "-o") {
      outFileName = av[++i];
    } else if (arg == "-q") {
      queryFileName = av[++i];
    } else
      throw std::runtime_error("./cuBQL_convert_br -o outFileName pathTo/inputFiles*");
  }
  if (outFileName.empty())
    throw std::runtime_error("no output filename specified (-o)");
  if (queryFileName.empty())
    throw std::runtime_error("no query filename specified (-q)");
  if (inFileNames.empty())
    throw std::runtime_error("no input file(s) specified");

  std::vector<float3> mergedPoints;
  std::vector<std::vector<float3>> blocks;
  for (auto fn : inFileNames) {
    std::vector<float3> points;
    std::cout << "importing from " << fn << " ..." << std::endl;
    std::ifstream in(fn,std::ios::binary);
    float2 offset;
    int numPoints;
    in.read((char *)&offset,sizeof(offset));
    in.read((char *)&numPoints,sizeof(numPoints));
    for (int i=0;i<numPoints;i++) {
      float3 p, n;
      in.read((char *)&p,sizeof(p));
      in.read((char *)&n,sizeof(n));
      p.x += offset.x;
      p.y += offset.y;
      points.push_back(p);
      mergedPoints.push_back(p);
    }
    std::cout << " num points now " << prettyNumber(mergedPoints.size()) << std::endl;;
    blocks.push_back(points);
  }
  std::cout << "saving merged..." << std::endl;
  saveData(mergedPoints,outFileName+"_merged_data.pts");
  std::cout << "saving blocks..." << std::endl;
  // saveData2(points,outFileName+"_data.bop");
  std::ofstream data(outFileName+"_data.bop",std::ios::binary);
  write(data,blocks);

  std::vector<float3> points;
  std::cout << "reading " << queryFileName << std::endl;
  std::ifstream in(queryFileName,std::ios::binary);
  int numPoints;
  in.read((char *)&numPoints,sizeof(numPoints));
  for (int i=0;i<numPoints;i++) {
    float3 p;
    in.read((char *)&p,sizeof(p));
    points.push_back(p);
  }
  saveData(points,outFileName+"_queries.pts");
  
  return 0;
}

