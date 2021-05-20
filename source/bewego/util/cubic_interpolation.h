/*
 * Copyright (c) 2021
 * All rights reserved.
 *
 * Redistribution  and  use  in  source  and binary  forms,  with  or  without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of  source  code must retain the  above copyright
 *      notice and this list of conditions.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice and  this list of  conditions in the  documentation and/or
 *      other materials provided with the distribution.
 *
 * THE SOFTWARE  IS PROVIDED "AS IS"  AND THE AUTHOR  DISCLAIMS ALL WARRANTIES
 * WITH  REGARD   TO  THIS  SOFTWARE  INCLUDING  ALL   IMPLIED  WARRANTIES  OF
 * MERCHANTABILITY AND  FITNESS.  IN NO EVENT  SHALL THE AUTHOR  BE LIABLE FOR
 * ANY  SPECIAL, DIRECT,  INDIRECT, OR  CONSEQUENTIAL DAMAGES  OR  ANY DAMAGES
 * WHATSOEVER  RESULTING FROM  LOSS OF  USE, DATA  OR PROFITS,  WHETHER  IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR  OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 *
 *                                              Jim Mainprice Wed 19 May 2021
 */

#pragma once

#include <Eigen/Dense>
#include <vector>

class CubicInterpolator {
 public:
  typedef double fptype;

  /* Initializes an interpolator regular interpolation */
  CubicInterpolator(const std::vector<fptype>& data, fptype spacing);
  ~CubicInterpolator();

  fptype Evaluate(double point) const;

 protected:
  std::vector<fptype> data_;
  fptype _spacing;
  Eigen::Matrix<fptype, 4, 4> A_;
  Eigen::Matrix<fptype, 4, 4> C_;
  Eigen::Matrix<fptype, 4, 1> coefs_;
};

class BiCubicGridInterpolator {
 public:
  typedef double fptype;

  /* Initializes an interpolator using a
     specified datacube of length n1 x n2,

     where data is ordered first along the n1 axis
     [0,0], [1,0], ..., [n1-1,0], [0,1], ... If n2 is
     omitted, then n1=n2=n3 is assumed.
     Data is assumed to be equally spaced and
     periodic along each axis, with the coordinate origin (0,0)
     at grid index [0,0].
   */
  BiCubicGridInterpolator(const std::vector<fptype>& data, fptype spacing,
                          int n1, int n2);
  ~BiCubicGridInterpolator();

  fptype Evaluate(const Eigen::Matrix<fptype, 2, 1>& point) const;

  double Interpolate(double p[4], double x);
  double Interpolate(double p[4][4], double x, double y);

 protected:
  std::vector<fptype> data_;
  fptype _spacing;
  int _n1, _n2;
  int _i1, _i2;
  bool _initialized;
  Eigen::Matrix<fptype, 16, 1> _coefs;
  inline int _index(int i1, int i2) const {
    if ((i1 %= _n1) < 0) i1 += _n1;
    if ((i2 %= _n2) < 0) i2 += _n2;
    return i1 + _n1 * i2;
  }
};

/* Tri-Cubic Interpolation

 This code is adapted from https://github.com/deepzot/likely
 Performs tri-cubic interpolation within a 3D periodic grid.

 Based on:

    Lekien, F., & Marsden, J. (2005).
    Tricubic interpolation in three dimensions.
    International Journal for Numerical Methods in Engineering, 63(3), 455-471.
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.89.7835
 */
class TriCubicGridInterpolator {
 public:
  typedef double fptype;

  /* Initializes an interpolator using a
     specified datacube of length n1 x n2 x n3,

     where data is ordered first along the n1 axis
     [0,0,0], [1,0,0], ..., [n1-1,0,0], [0,1,0], ... If n2 and n3 are both
     omitted, then n1=n2=n3 is assumed. Data is assumed to be equally spaced and
     periodic along each axis, with the coordinate origin (0,0,0)
     at grid index [0,0,0].
   */
  TriCubicGridInterpolator(const std::vector<fptype>& data, fptype spacing,
                           int n1, int n2, int n3);
  ~TriCubicGridInterpolator();

  fptype Evaluate(const Eigen::Matrix<fptype, 3, 1>& point);

 protected:
  /* Initialize the Data for voxel xi, yi, zi */
  Eigen::Matrix<fptype, 64, 1> FiniteDifferenceDataAtCorners(int xi, int yi,
                                                             int zi) const;

  /* Initialize the C matrix (Called on construction) */
  void InitializeC1Matrix();
  void InitializeC2Matrix();

  std::vector<fptype> data_;
  fptype _spacing;
  int _n1, _n2, _n3;
  int _i1, _i2, _i3;
  bool _initialized;
  Eigen::Matrix<fptype, 64, 1> _coefs;
  Eigen::Matrix<fptype, 64, 64> _C1;  // _C1 = A_inv matrix
  Eigen::Matrix<fptype, 64, 64> _C2;  // _C2 = A_inv * B : Matrix including FD
  inline int _index(int i1, int i2, int i3) const {
    if ((i1 %= _n1) < 0) i1 += _n1;
    if ((i2 %= _n2) < 0) i2 += _n2;
    if ((i3 %= _n3) < 0) i3 += _n3;
    return i1 + _n1 * (i2 + _n2 * i3);
  }
};
