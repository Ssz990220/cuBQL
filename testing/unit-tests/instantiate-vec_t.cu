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

#include "cuBQL/common/vec.h"

using namespace cuBQL;

template<typename T, int D>
void foo()
{
  using vec_t = cuBQL::vec_t<T,D>;

  vec_t a = make<vec_t>((T)1);
  vec_t b = make<vec_t>((T)2);
  // __attribute__((unused))
  typename dot_result_t<T>::type d = dot(a,b);
}

int main(int, char **)
{
  foo<float,2>();
  foo<float,3>();
  foo<float,4>();
  foo<float,CUBQL_TEST_N>();
  return 0;
}

