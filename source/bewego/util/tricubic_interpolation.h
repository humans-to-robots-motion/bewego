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

// This code is adapted from https://github.com/deepzot/likely
// Performs tri-cubic interpolation within a 3D periodic grid.
// Based on http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.89.7835
class TriCubicGridInterpolator {
 public:
  typedef double fptype;

  TriCubicGridInterpolator(const std::vector<fptype>& data, fptype spacing,
                           int n1, int n2, int n3);
  ~TriCubicGridInterpolator();

  fptype Evaluate(const Eigen::Matrix<fptype, 3, 1>& point);

 private:
  std::vector<fptype> data_;
  fptype _spacing;
  int _n1, _n2, _n3;
  int _i1, _i2, _i3;
  bool _initialized;
  Eigen::Matrix<fptype, 64, 1> _coefs;
  Eigen::Matrix<fptype, 64, 64> _C;
  inline int _index(int i1, int i2, int i3) const {
    if ((i1 %= _n1) < 0) i1 += _n1;
    if ((i2 %= _n2) < 0) i2 += _n2;
    if ((i3 %= _n3) < 0) i3 += _n3;
    return i1 + _n1 * (i2 + _n2 * i3);
  }
};
