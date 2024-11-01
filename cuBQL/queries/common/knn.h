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

/*! \file cuBQL/queries/common/knn.h Helper tools for knn-style queries

  Different types of kNN queries may have differnet ways of computing
  distance between query and data primitmives, and different BVHes may
  have different traversal routines - but all types of knns require
  some form of "candidate management" - ie, tracking how many (and
  which) possible k nearest candidates have already been found,
  inserting or evicting candidates from that list, etc. This
  functionality is indepenent of data type, and should only be written
  once
 */

#pragma once

#include "cuBQL/bvh.h"

namespace cuBQL {
  namespace knn {
    
    /*! by default we return the *number* of found candidates (within
        the query radius), and the (square) distance to the most
        distant one, or (square) max query radius if less than the
        requested num prims were found */
    struct Result {
      /*! number of kNNs found */
      int numFound;
      /*! (square) distance to most distant found candidate, or
          INFINITY if numFound is 0 */
      float sqrDistMax;
    };

    /*! one of the cancidates in a kNN candidate list; this refers to
        a data primitive, and the (square) distance to that candidate
        primitive that the respective data type has computed for the
        given query primitive */
    struct Candidate {
      int   primID;
      float sqrDist;
    };

    inline __cubql_both
    void insert_linear(Candidate *candidates,
                       int        maxCandidates,
                       Candidate  newCandidate,
                       Result    &result)
    {
      int insertPos;
      if (result.numFound < maxCandidates) {
        insertPos = result.numFound++;
      } else {
        if (newCandidate.sqrDist >= result.sqrDistMax)
          return;
        insertPos = maxCandidates-1;
      }
      while (insertPos > 0 && candidates[insertPos-1].sqrDist > newCandidate.sqrDist) {
        candidates[insertPos] = candidates[insertPos-1];
        --insertPos;
      }
      candidates[insertPos] = newCandidate;
      if (result.numFound == maxCandidates)
        result.sqrDistMax = candidates[maxCandidates-1].sqrDist;
    }
    
  } // ::cuBQL::knn
} // ::cuBQL
