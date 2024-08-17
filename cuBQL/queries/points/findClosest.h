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

/*! \file cuBQL/queries/points/findClosest Provides kernels for
  finding closest point(s) on point data */

#pragma once

#include "cuBQL/bvh.h"

namespace cuBQL {
  namespace points {

    // ******************************************************************
    // INTERFACE
    // (which functions this header file provides)
    // ******************************************************************
    
    /*! given a bvh build over a set of float<N> points, perform a
      closest-point query that returns the index of the input point
      closest to the query point (if one exists within the given max
      query radius), or -1 (if not). 

      \returns Index of point in points[] array that is closest to
      query point, or -1 if no point exists within provided max query
      range

      \note If more than one point with similar closest distance
      exist, then this function will not make any guarantees as to
      which of them will be returned (though we can expect that
      succesuve such queries on the _same_ bvh will return the same
      result, different BVHs built even over the same input data may
      not)
    */
    inline __device__
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    BinaryBVH<float,D> bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const vec_t<float,D> *points,
                    /*! the query point for which we want to know the
                      result */
                    vec_t<float,D> queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=INFINITY);

    inline __device__
    int findClosest_exludeID(/*! primitive ID to _exclude_ from queries */,
                             int idOfPointtoExclude,
                             /*! binary bvh built over the given points[]
                               specfied below */
                             BinaryBVH<float,D> bvhOverPoints,
                             /*! data points that the bvh was built over */
                             const vec_t<float,D> *points,
                             /*! the query point for which we want to know the
                               result */
                             vec_t<float,D> queryPoint,
                             /*! square of the maximum query distance in which
                               this query is to look for candidates. note
                               this is the SQUARE distance */
                             float squareOfMaxQueryDistance=INFINITY);

    /*! same as regular points::closestPoint, but excluding all data
      points that are at the query position itself */ 
    inline __device__
    int findClosest_exludeSelf(/*! binary bvh built over the given points[]
                                 specfied below */
                               BinaryBVH<float,D> bvhOverPoints,
                               /*! data points that the bvh was built over */
                               const vec_t<float,D> *points,
                               /*! the query point for which we want to know the
                                 result */
                               vec_t<float,D> queryPoint,
                               /*! square of the maximum query distance in which
                                 this query is to look for candidates. note
                                 this is the SQUARE distance */
                               float squareOfMaxQueryDistance=INFINITY);
    


    
    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float2 type, just for convenience */
    inline __device__
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float2 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float2 *points,
                    float2 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=INFINITY)

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float3 type, just for convenience */
      inline __device__
      int findClosest(/*! binary bvh built over the given points[]
                        specfied below */
                      bvh_float3 bvhOverPoints,
                      /*! data points that the bvh was built over */
                      const float3 *points,
                      float3 queryPoint,
                      /*! square of the maximum query distance in which
                        this query is to look for candidates. note
                        this is the SQUARE distance */
                      float squareOfMaxQueryDistance=INFINITY);

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float4 type, just for convenience */
    inline __device__
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float4 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float4 *points,
                    float4 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=INFINITY);

    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************

    template<typename BlackListLambda>
    inline __device__
    int findClosest_withBlackList(const BlackListLambda blackListed,
                                  /*! binary bvh built over the given points[]
                                    specfied below */
                                  BinaryBVH<float,D> bvhOverPoints,
                                  /*! data points that the bvh was built over */
                                  const vec_t<float,D> *points,
                                  /*! the query point for which we want to know the
                                    result */
                                  vec_t<float,D> queryPoint,
                                  /*! square of the maximum query distance in which
                                    this query is to look for candidates. note
                                    this is the SQUARE distance */
                                  float squareOfMaxQueryDistance=INFINITY)
    {
      int closestID = -1;
      float closestSqrDist = squareOfMaxQueryDistance;
      // callback that processes each candidate, and checks if its
      // closer than current best
      auto candidateLambda = [blackListed,closestID,closestSqrDist](int pointID)->bool {
        if (blackListed(pointID))
          // caller explicitly blacklisted this point, do not process
          return closestSqrDist;

        // compute (square distance)
        float sqrDist = sqrDistance(points[pointID],queryPoint);
        if (sqrDist >= closestSqrDist)
          // candidate is further away than what we already have
          return closestSqrDist;

        // candidate is closer - accept and update search distance
        closestSqrDist = sqrDist;
        closestID      = pointID;
        return closestSqrDist;
      };

      cuBQL::shrinkingRadiusQuery_forEachPrim(bvh,queryPoint,
                                              squareOfMaxQueryDistance,
                                              candidateLambda);
      return closestID;
    }
    
    inline __device__
    int findClosest_exludeID(/*! primitive ID to _exclude_ from queries */,
                             int idOfPointtoExclude,
                             /*! binary bvh built over the given points[]
                               specfied below */
                             BinaryBVH<float,D> bvhOverPoints,
                             /*! data points that the bvh was built over */
                             const vec_t<float,D> *points,
                             /*! the query point for which we want to know the
                               result */
                             vec_t<float,D> queryPoint,
                             /*! square of the maximum query distance in which
                               this query is to look for candidates. note
                               this is the SQUARE distance */
                             float squareOfMaxQueryDistance=INFINITY)
    {
      /* blacklist the ID itself, then call
         `findClosest_withBlackList()` */
      auto blackList = [](int pointID)->bool {
        return pointID == idOfPointToExclude;
      };
      findClosest_withBlackList(blackList,
                                bvhOverPoints,
                                points,
                                queryPoint,
                                squareOfMaxQueryDistance);
    }

    /*! same as regular points::closestPoint, but excluding all data
      points that are at the query position itself */ 
    inline __device__
    int findClosest_exludeSelf(/*! binary bvh built over the given points[]
                                 specfied below */
                               BinaryBVH<float,D> bvhOverPoints,
                               /*! data points that the bvh was built over */
                               const vec_t<float,D> *points,
                               /*! the query point for which we want to know the
                                 result */
                               vec_t<float,D> queryPoint,
                               /*! square of the maximum query distance in which
                                 this query is to look for candidates. note
                                 this is the SQUARE distance */
                               float squareOfMaxQueryDistance=INFINITY)
    {
      /* blacklist any point at same position as query point, then
         call `findClosest_withBlackList()` */
      auto blackList = [](int pointID)->bool {
        return points[pointID] == queryPoint;
      };
      findClosest_withBlackList(blackList,
                                bvhOverPoints,
                                points,
                                queryPoint,
                                squareOfMaxQueryDistance);
    }

    
    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float2 type, just for convenience */
    inline __device__
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float2 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float2 *points,
                    float2 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    { return findClosest(bvhOverPoints,
                         (const vec_t<float,2> *)points,
                         (const vec_t<float,2> &)queryPoint,
                         squareOfMaxQueryDistance);
    }

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float3 type, just for convenience */
    inline __device__
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float3 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float3 *points,
                    float3 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    { return findClosest(bvhOverPoints,
                         (const vec_t<float,3> *)points,
                         (const vec_t<float,3> &)queryPoint,
                         squareOfMaxQueryDistance);
    }

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float4 type, just for convenience */
    inline __device__
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float4 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float4 *points,
                    float4 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    { return findClosest(bvhOverPoints,
                         (const vec_t<float,4> *)points,
                         (const vec_t<float,4> &)queryPoint,
                         squareOfMaxQueryDistance);
    }
  }
}
