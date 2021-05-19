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
// This code is adapted from
// https://github.com/danielguterding/pytricubic/blob/master/src/tricubic.cpp

#include <bewego/util/tricubic_interpolation.h>

#include <stdexcept>
#include <string>

/*
 Initializes an interpolator using the specified datacube of length n1 x n2 x n3
 where data is ordered first along the n1 axis [0,0,0], [1,0,0], ...,
 [n1-1,0,0], [0,1,0], ... If n2 and n3 are both omitted, then n1=n2=n3
 is assumed. Data is assumed to be equally spaced and periodic along
 each axis, with the coordinate origin (0,0,0) at grid index [0,0,0].
 */
TriCubicGridInterpolator::TriCubicGridInterpolator(
    const std::vector<fptype>& data, fptype spacing, int n1, int n2, int n3) {
  _initialized = false;
  _spacing = 1.0;  // = spacing
  _n1 = n1;
  _n2 = n2;
  _n3 = n3;
  data_ = data;

  if (_n2 == 0 && _n3 == 0) {
    _n3 = _n2 = _n1;
  }
  if (_n1 <= 0 || _n2 <= 0 || _n3 <= 0) {
    throw std::runtime_error("Bad datacube dimensions.");
  }
  if (_spacing <= 0) {
    throw std::runtime_error("Bad datacube grid spacing.");
  }

  // temporary array is necessary, otherwise compiler has problems with Eigen
  // and takes very long to compile
  const int temp[64][64] = {
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, -3, 3, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0,  0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0},
      {-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0},
      {0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 3,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0,  0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0},
      {9, -9, -9, 9, 0, 0, 0, 0, 6, 3, -6, -3, 0, 0, 0, 0, 6, -6, 3, -3, 0, 0,
       0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 4,  2,  2, 1, 0, 0, 0, 0,  0, 0,  0, 0,
       0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,  0, 0},
      {-6, 6,  6,  -6, 0, 0, 0, 0, -3, -3, 3, 3, 0, 0, 0, 0,
       -4, 4,  -2, 2,  0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,
       -2, -2, -1, -1, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,
       0,  0,  0,  0,  0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0},
      {2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0},
      {-6, 6,  6,  -6, 0, 0, 0, 0, -4, -2, 4, 2, 0, 0, 0, 0,
       -3, 3,  -3, 3,  0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,
       -2, -1, -2, -1, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,
       0,  0,  0,  0,  0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0},
      {4, -4, -4, 4, 0, 0, 0, 0, 2, 2, -2, -2, 0, 0, 0, 0, 2, -2, 2, -2, 0, 0,
       0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 1,  1,  1, 1, 0, 0, 0, 0,  0, 0,  0, 0,
       0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,  0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0,
       0, 0, -3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0,
       0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0},
      {0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
       0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -3, 3, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0},
      {0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, -3, 0, 3,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0,  0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0,  0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, -3, 0, 3, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0,  0},
      {0, 0, 0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0, 0, 9, -9, -9, 9,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, -6, -3,
       0, 0, 0, 0,  6,  -6, 3, -3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0},
      {0,  0, 0,  0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0,
       0,  0, 0,  0, 0, 0, 0, 0, -6, 6,  6,  -6, 0, 0, 0, 0,
       0,  0, 0,  0, 0, 0, 0, 0, -3, -3, 3,  3,  0, 0, 0, 0,
       -4, 4, -2, 2, 0, 0, 0, 0, -2, -2, -1, -1, 0, 0, 0, 0},
      {0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1,  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0},
      {0,  0, 0,  0, 0, 0, 0, 0, 0,  0,  0,  0,  0, 0, 0, 0,
       0,  0, 0,  0, 0, 0, 0, 0, -6, 6,  6,  -6, 0, 0, 0, 0,
       0,  0, 0,  0, 0, 0, 0, 0, -4, -2, 4,  2,  0, 0, 0, 0,
       -3, 3, -3, 3, 0, 0, 0, 0, -2, -1, -2, -1, 0, 0, 0, 0},
      {0, 0, 0, 0,  0,  0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0, 0, 4, -4, -4, 4,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, -2, -2,
       0, 0, 0, 0,  2,  -2, 2, -2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0},
      {-3, 0, 0,  0, 3, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0,  0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0,  0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0,  0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,  0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0,
       -1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0},
      {9, -9, 0, 0,  -9, 9, 0, 0,  6, 3, 0, 0, -6, -3, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0,  6, -6, 0,  0, 3, -3, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 4, 2, 0, 0,
       2, 1,  0, 0,  0,  0, 0, 0,  0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0},
      {-6, 6, 0, 0, 6, -6, 0, 0, -3, -3, 0, 0, 3,  3,  0, 0,
       0,  0, 0, 0, 0, 0,  0, 0, -4, 4,  0, 0, -2, 2,  0, 0,
       0,  0, 0, 0, 0, 0,  0, 0, -2, -2, 0, 0, -1, -1, 0, 0,
       0,  0, 0, 0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0},
      {0, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 3, 0,
       0, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
       0, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0,  0, 3, 0, 0,  0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, -2, 0, 0, 0, -1, 0, 0, 0},
      {0, 0, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0,  0,  9, -9, 0, 0, -9, 9,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 0,  6, 3, 0, 0, -6, -3, 0, 0,  0, 0, 0,  0,
       0, 0, 0, 0, 6, -6, 0, 0, 3, -3, 0, 0, 4, 2, 0,  0,  2, 1,  0, 0},
      {0,  0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0,
       -6, 6,  0, 0, 6,  -6, 0, 0, 0,  0,  0, 0, 0,  0,  0, 0,
       -3, -3, 0, 0, 3,  3,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0,
       -4, 4,  0, 0, -2, 2,  0, 0, -2, -2, 0, 0, -1, -1, 0, 0},
      {9,  0, -9, 0, -9, 0, 9, 0, 0,  0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0, -6, 0,
       -3, 0, 6,  0, -6, 0, 3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0,  0, 0,  0, 4,  0, 2, 0, 2,  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0,  0, 0, 0, 0, 0, 9, 0, -9, 0, -9, 0, 9,  0, 0,  0, 0, 0, 0,  0,
       0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 6,  0, 3,  0, -6, 0, -3, 0, 6, 0, -6, 0,
       3, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0,  0, 4,  0, 2,  0, 2,  0, 1, 0},
      {-27, 27, 27, -27, 27, -27, -27, 27, -18, -9, 18, 9,   18, 9,  -18, -9,
       -18, 18, -9, 9,   18, -18, 9,   -9, -18, 18, 18, -18, -9, 9,  9,   -9,
       -12, -6, -6, -3,  12, 6,   6,   3,  -12, -6, 12, 6,   -6, -3, 6,   3,
       -12, 12, -6, 6,   -6, 6,   -3,  3,  -8,  -4, -4, -2,  -4, -2, -2,  -1},
      {18, -18, -18, 18, -18, 18, 18, -18, 9,  9,   -9,  -9, -9, -9, 9,  9,
       12, -12, 6,   -6, -12, 12, -6, 6,   12, -12, -12, 12, 6,  -6, -6, 6,
       6,  6,   3,   3,  -6,  -6, -3, -3,  6,  6,   -6,  -6, 3,  3,  -3, -3,
       8,  -8,  4,   -4, 4,   -4, 2,  -2,  4,  4,   2,   2,  2,  2,  1,  1},
      {-6, 0, 6,  0, 6,  0, -6, 0, 0,  0, 0,  0, 0, 0, 0, 0, -3, 0, -3, 0, 3, 0,
       3,  0, -4, 0, 4,  0, -2, 0, 2,  0, 0,  0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
       0,  0, 0,  0, -2, 0, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0,  0, 0,  0},
      {0,  0, 0, 0, 0, 0, 0, 0, -6, 0, 6,  0, 6,  0, -6, 0, 0,  0, 0,  0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0,  0, -3, 0, -3, 0, 3,  0, 3,  0, -4, 0, 4, 0,
       -2, 0, 2, 0, 0, 0, 0, 0, 0,  0, 0,  0, -2, 0, -2, 0, -1, 0, -1, 0},
      {18, -18, -18, 18, -18, 18, 18, -18, 12, 6,   -12, -6, -12, -6, 12, 6,
       9,  -9,  9,   -9, -9,  9,  -9, 9,   12, -12, -12, 12, 6,   -6, -6, 6,
       6,  3,   6,   3,  -6,  -3, -6, -3,  8,  4,   -8,  -4, 4,   2,  -4, -2,
       6,  -6,  6,   -6, 3,   -3, 3,  -3,  4,  2,   4,   2,  2,   1,  2,  1},
      {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6,  6,  6,  6,  -6, -6,
       -6,  6,  -6, 6,   6,  -6,  6,   -6, -8, 8,  8,  -8, -4, 4,  4,  -4,
       -3,  -3, -3, -3,  3,  3,   3,   3,  -4, -4, 4,  4,  -2, -2, 2,  2,
       -4,  4,  -4, 4,   -2, 2,   -2,  2,  -2, -2, -2, -2, -1, -1, -1, -1},
      {2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 1, 0, 0,  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 1, 0, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0},
      {-6, 6, 0, 0, 6, -6, 0, 0, -4, -2, 0, 0, 4,  2,  0, 0,
       0,  0, 0, 0, 0, 0,  0, 0, -3, 3,  0, 0, -3, 3,  0, 0,
       0,  0, 0, 0, 0, 0,  0, 0, -2, -1, 0, 0, -2, -1, 0, 0,
       0,  0, 0, 0, 0, 0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0},
      {4, -4, 0, 0,  -4, 4, 0, 0,  2, 2, 0, 0, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0,  2, -2, 0,  0, 2, -2, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 1, 1, 0, 0,
       1, 1,  0, 0,  0,  0, 0, 0,  0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, 1, 0, 0, 0},
      {0,  0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0,
       -6, 6,  0, 0, 6,  -6, 0, 0, 0,  0,  0, 0, 0,  0,  0, 0,
       -4, -2, 0, 0, 4,  2,  0, 0, 0,  0,  0, 0, 0,  0,  0, 0,
       -3, 3,  0, 0, -3, 3,  0, 0, -2, -1, 0, 0, -2, -1, 0, 0},
      {0, 0, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0,  0,  4, -4, 0, 0, -4, 4,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 0,  2, 2, 0, 0, -2, -2, 0, 0,  0, 0, 0,  0,
       0, 0, 0, 0, 2, -2, 0, 0, 2, -2, 0, 0, 1, 1, 0,  0,  1, 1,  0, 0},
      {-6, 0, 6,  0, 6,  0, -6, 0, 0,  0, 0,  0, 0, 0, 0, 0, -4, 0, -2, 0, 4, 0,
       2,  0, -3, 0, 3,  0, -3, 0, 3,  0, 0,  0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
       0,  0, 0,  0, -2, 0, -1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0,  0, 0,  0},
      {0,  0, 0, 0, 0, 0, 0, 0, -6, 0, 6,  0, 6,  0, -6, 0, 0,  0, 0,  0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0,  0, -4, 0, -2, 0, 4,  0, 2,  0, -3, 0, 3, 0,
       -3, 0, 3, 0, 0, 0, 0, 0, 0,  0, 0,  0, -2, 0, -1, 0, -2, 0, -1, 0},
      {18, -18, -18, 18, -18, 18, 18, -18, 12, 6,  -12, -6, -12, -6, 12, 6,
       12, -12, 6,   -6, -12, 12, -6, 6,   9,  -9, -9,  9,  9,   -9, -9, 9,
       8,  4,   4,   2,  -8,  -4, -4, -2,  6,  3,  -6,  -3, 6,   3,  -6, -3,
       6,  -6,  3,   -3, 6,   -6, 3,  -3,  4,  2,  2,   1,  4,   2,  2,  1},
      {-12, 12, 12, -12, 12, -12, -12, 12, -6, -6, 6,  6,  6,  6,  -6, -6,
       -8,  8,  -4, 4,   8,  -8,  4,   -4, -6, 6,  6,  -6, -6, 6,  6,  -6,
       -4,  -4, -2, -2,  4,  4,   2,   2,  -3, -3, 3,  3,  -3, -3, 3,  3,
       -4,  4,  -2, 2,   -4, 4,   -2,  2,  -2, -2, -1, -1, -2, -2, -1, -1},
      {4,  0, -4, 0, -4, 0, 4, 0, 0,  0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, -2, 0,
       -2, 0, 2,  0, -2, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0,  0, 0,  0, 1,  0, 1, 0, 1,  0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0,  0, 0, 0, 0, 0, 4, 0, -4, 0, -4, 0, 4,  0, 0,  0, 0, 0, 0,  0,
       0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 2,  0, 2,  0, -2, 0, -2, 0, 2, 0, -2, 0,
       2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1,  0, 1,  0, 1,  0, 1, 0},
      {-12, 12, 12, -12, 12, -12, -12, 12, -8, -4, 8,  4,  8,  4,  -8, -4,
       -6,  6,  -6, 6,   6,  -6,  6,   -6, -6, 6,  6,  -6, -6, 6,  6,  -6,
       -4,  -2, -4, -2,  4,  2,   4,   2,  -4, -2, 4,  2,  -4, -2, 4,  2,
       -3,  3,  -3, 3,   -3, 3,   -3,  3,  -2, -1, -2, -1, -2, -1, -2, -1},
      {8, -8, -8, 8,  -8, 8,  8,  -8, 4, 4,  -4, -4, -4, -4, 4,  4,
       4, -4, 4,  -4, -4, 4,  -4, 4,  4, -4, -4, 4,  4,  -4, -4, 4,
       2, 2,  2,  2,  -2, -2, -2, -2, 2, 2,  -2, -2, 2,  2,  -2, -2,
       2, -2, 2,  -2, 2,  -2, 2,  -2, 1, 1,  1,  1,  1,  1,  1,  1}};

  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++) _C(i, j) = temp[i][j];
}

TriCubicGridInterpolator::~TriCubicGridInterpolator() {}

TriCubicGridInterpolator::fptype TriCubicGridInterpolator::Evaluate(
    const Eigen::Matrix<fptype, 3, 1>& point) {
  fptype x = point.x();
  fptype y = point.y();
  fptype z = point.z();

  fptype dx = fmod(x / _spacing, _n1);
  fptype dy = fmod(y / _spacing, _n2);
  fptype dz = fmod(z / _spacing, _n3);
  // determine the relative position in the
  // box enclosed by nearest data points

  if (dx < 0) dx += _n1;  // periodicity is built in
  if (dy < 0) dy += _n2;
  if (dz < 0) dz += _n3;

  int xi = (int)floor(dx);  // calculate lower-bound grid indices
  int yi = (int)floor(dy);
  int zi = (int)floor(dz);

  // Check if we can re-use coefficients from the last interpolation.
  if (!_initialized || xi != _i1 || yi != _i2 || zi != _i3) {
    // Extract the local vocal values and calculate partial derivatives.
    Eigen::Matrix<fptype, 64, 1> x;
    x <<
        // values of f(x,y,z) at each corner.
        data_[_index(xi, yi, zi)],
        data_[_index(xi + 1, yi, zi)], data_[_index(xi, yi + 1, zi)],
        data_[_index(xi + 1, yi + 1, zi)], data_[_index(xi, yi, zi + 1)],
        data_[_index(xi + 1, yi, zi + 1)], data_[_index(xi, yi + 1, zi + 1)],
        data_[_index(xi + 1, yi + 1, zi + 1)],
        // values of df/dx at each corner.
        0.5 * (data_[_index(xi + 1, yi, zi)] - data_[_index(xi - 1, yi, zi)]),
        0.5 * (data_[_index(xi + 2, yi, zi)] - data_[_index(xi, yi, zi)]),
        0.5 * (data_[_index(xi + 1, yi + 1, zi)] -
               data_[_index(xi - 1, yi + 1, zi)]),
        0.5 *
            (data_[_index(xi + 2, yi + 1, zi)] - data_[_index(xi, yi + 1, zi)]),
        0.5 * (data_[_index(xi + 1, yi, zi + 1)] -
               data_[_index(xi - 1, yi, zi + 1)]),
        0.5 *
            (data_[_index(xi + 2, yi, zi + 1)] - data_[_index(xi, yi, zi + 1)]),
        0.5 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
               data_[_index(xi - 1, yi + 1, zi + 1)]),
        0.5 * (data_[_index(xi + 2, yi + 1, zi + 1)] -
               data_[_index(xi, yi + 1, zi + 1)]),
        // values of df/dy at each corner.
        0.5 * (data_[_index(xi, yi + 1, zi)] - data_[_index(xi, yi - 1, zi)]),
        0.5 * (data_[_index(xi + 1, yi + 1, zi)] -
               data_[_index(xi + 1, yi - 1, zi)]),
        0.5 * (data_[_index(xi, yi + 2, zi)] - data_[_index(xi, yi, zi)]),
        0.5 *
            (data_[_index(xi + 1, yi + 2, zi)] - data_[_index(xi + 1, yi, zi)]),
        0.5 * (data_[_index(xi, yi + 1, zi + 1)] -
               data_[_index(xi, yi - 1, zi + 1)]),
        0.5 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
               data_[_index(xi + 1, yi - 1, zi + 1)]),
        0.5 *
            (data_[_index(xi, yi + 2, zi + 1)] - data_[_index(xi, yi, zi + 1)]),
        0.5 * (data_[_index(xi + 1, yi + 2, zi + 1)] -
               data_[_index(xi + 1, yi, zi + 1)]),
        // values of df/dz at each corner.
        0.5 * (data_[_index(xi, yi, zi + 1)] - data_[_index(xi, yi, zi - 1)]),
        0.5 * (data_[_index(xi + 1, yi, zi + 1)] -
               data_[_index(xi + 1, yi, zi - 1)]),
        0.5 * (data_[_index(xi, yi + 1, zi + 1)] -
               data_[_index(xi, yi + 1, zi - 1)]),
        0.5 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
               data_[_index(xi + 1, yi + 1, zi - 1)]),
        0.5 * (data_[_index(xi, yi, zi + 2)] - data_[_index(xi, yi, zi)]),
        0.5 *
            (data_[_index(xi + 1, yi, zi + 2)] - data_[_index(xi + 1, yi, zi)]),
        0.5 *
            (data_[_index(xi, yi + 1, zi + 2)] - data_[_index(xi, yi + 1, zi)]),
        0.5 * (data_[_index(xi + 1, yi + 1, zi + 2)] -
               data_[_index(xi + 1, yi + 1, zi)]),
        // values of d2f/dxdy at each corner.
        0.25 * (data_[_index(xi + 1, yi + 1, zi)] -
                data_[_index(xi - 1, yi + 1, zi)] -
                data_[_index(xi + 1, yi - 1, zi)] +
                data_[_index(xi - 1, yi - 1, zi)]),
        0.25 *
            (data_[_index(xi + 2, yi + 1, zi)] - data_[_index(xi, yi + 1, zi)] -
             data_[_index(xi + 2, yi - 1, zi)] + data_[_index(xi, yi - 1, zi)]),
        0.25 * (data_[_index(xi + 1, yi + 2, zi)] -
                data_[_index(xi - 1, yi + 2, zi)] -
                data_[_index(xi + 1, yi, zi)] + data_[_index(xi - 1, yi, zi)]),
        0.25 *
            (data_[_index(xi + 2, yi + 2, zi)] - data_[_index(xi, yi + 2, zi)] -
             data_[_index(xi + 2, yi, zi)] + data_[_index(xi, yi, zi)]),
        0.25 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
                data_[_index(xi - 1, yi + 1, zi + 1)] -
                data_[_index(xi + 1, yi - 1, zi + 1)] +
                data_[_index(xi - 1, yi - 1, zi + 1)]),
        0.25 * (data_[_index(xi + 2, yi + 1, zi + 1)] -
                data_[_index(xi, yi + 1, zi + 1)] -
                data_[_index(xi + 2, yi - 1, zi + 1)] +
                data_[_index(xi, yi - 1, zi + 1)]),
        0.25 * (data_[_index(xi + 1, yi + 2, zi + 1)] -
                data_[_index(xi - 1, yi + 2, zi + 1)] -
                data_[_index(xi + 1, yi, zi + 1)] +
                data_[_index(xi - 1, yi, zi + 1)]),
        0.25 *
            (data_[_index(xi + 2, yi + 2, zi + 1)] -
             data_[_index(xi, yi + 2, zi + 1)] -
             data_[_index(xi + 2, yi, zi + 1)] + data_[_index(xi, yi, zi + 1)]),
        // values of d2f/dxdz at each corner.
        0.25 * (data_[_index(xi + 1, yi, zi + 1)] -
                data_[_index(xi - 1, yi, zi + 1)] -
                data_[_index(xi + 1, yi, zi - 1)] +
                data_[_index(xi - 1, yi, zi - 1)]),
        0.25 *
            (data_[_index(xi + 2, yi, zi + 1)] - data_[_index(xi, yi, zi + 1)] -
             data_[_index(xi + 2, yi, zi - 1)] + data_[_index(xi, yi, zi - 1)]),
        0.25 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
                data_[_index(xi - 1, yi + 1, zi + 1)] -
                data_[_index(xi + 1, yi + 1, zi - 1)] +
                data_[_index(xi - 1, yi + 1, zi - 1)]),
        0.25 * (data_[_index(xi + 2, yi + 1, zi + 1)] -
                data_[_index(xi, yi + 1, zi + 1)] -
                data_[_index(xi + 2, yi + 1, zi - 1)] +
                data_[_index(xi, yi + 1, zi - 1)]),
        0.25 * (data_[_index(xi + 1, yi, zi + 2)] -
                data_[_index(xi - 1, yi, zi + 2)] -
                data_[_index(xi + 1, yi, zi)] + data_[_index(xi - 1, yi, zi)]),
        0.25 *
            (data_[_index(xi + 2, yi, zi + 2)] - data_[_index(xi, yi, zi + 2)] -
             data_[_index(xi + 2, yi, zi)] + data_[_index(xi, yi, zi)]),
        0.25 * (data_[_index(xi + 1, yi + 1, zi + 2)] -
                data_[_index(xi - 1, yi + 1, zi + 2)] -
                data_[_index(xi + 1, yi + 1, zi)] +
                data_[_index(xi - 1, yi + 1, zi)]),
        0.25 *
            (data_[_index(xi + 2, yi + 1, zi + 2)] -
             data_[_index(xi, yi + 1, zi + 2)] -
             data_[_index(xi + 2, yi + 1, zi)] + data_[_index(xi, yi + 1, zi)]),
        // values of d2f/dydz at each corner.
        0.25 * (data_[_index(xi, yi + 1, zi + 1)] -
                data_[_index(xi, yi - 1, zi + 1)] -
                data_[_index(xi, yi + 1, zi - 1)] +
                data_[_index(xi, yi - 1, zi - 1)]),
        0.25 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
                data_[_index(xi + 1, yi - 1, zi + 1)] -
                data_[_index(xi + 1, yi + 1, zi - 1)] +
                data_[_index(xi + 1, yi - 1, zi - 1)]),
        0.25 *
            (data_[_index(xi, yi + 2, zi + 1)] - data_[_index(xi, yi, zi + 1)] -
             data_[_index(xi, yi + 2, zi - 1)] + data_[_index(xi, yi, zi - 1)]),
        0.25 * (data_[_index(xi + 1, yi + 2, zi + 1)] -
                data_[_index(xi + 1, yi, zi + 1)] -
                data_[_index(xi + 1, yi + 2, zi - 1)] +
                data_[_index(xi + 1, yi, zi - 1)]),
        0.25 * (data_[_index(xi, yi + 1, zi + 2)] -
                data_[_index(xi, yi - 1, zi + 2)] -
                data_[_index(xi, yi + 1, zi)] + data_[_index(xi, yi - 1, zi)]),
        0.25 * (data_[_index(xi + 1, yi + 1, zi + 2)] -
                data_[_index(xi + 1, yi - 1, zi + 2)] -
                data_[_index(xi + 1, yi + 1, zi)] +
                data_[_index(xi + 1, yi - 1, zi)]),
        0.25 *
            (data_[_index(xi, yi + 2, zi + 2)] - data_[_index(xi, yi, zi + 2)] -
             data_[_index(xi, yi + 2, zi)] + data_[_index(xi, yi, zi)]),
        0.25 *
            (data_[_index(xi + 1, yi + 2, zi + 2)] -
             data_[_index(xi + 1, yi, zi + 2)] -
             data_[_index(xi + 1, yi + 2, zi)] + data_[_index(xi + 1, yi, zi)]),
        // values of d3f/dxdydz at each corner.
        0.125 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
                 data_[_index(xi - 1, yi + 1, zi + 1)] -
                 data_[_index(xi + 1, yi - 1, zi + 1)] +
                 data_[_index(xi - 1, yi - 1, zi + 1)] -
                 data_[_index(xi + 1, yi + 1, zi - 1)] +
                 data_[_index(xi - 1, yi + 1, zi - 1)] +
                 data_[_index(xi + 1, yi - 1, zi - 1)] -
                 data_[_index(xi - 1, yi - 1, zi - 1)]),
        0.125 * (data_[_index(xi + 2, yi + 1, zi + 1)] -
                 data_[_index(xi, yi + 1, zi + 1)] -
                 data_[_index(xi + 2, yi - 1, zi + 1)] +
                 data_[_index(xi, yi - 1, zi + 1)] -
                 data_[_index(xi + 2, yi + 1, zi - 1)] +
                 data_[_index(xi, yi + 1, zi - 1)] +
                 data_[_index(xi + 2, yi - 1, zi - 1)] -
                 data_[_index(xi, yi - 1, zi - 1)]),
        0.125 * (data_[_index(xi + 1, yi + 2, zi + 1)] -
                 data_[_index(xi - 1, yi + 2, zi + 1)] -
                 data_[_index(xi + 1, yi, zi + 1)] +
                 data_[_index(xi - 1, yi, zi + 1)] -
                 data_[_index(xi + 1, yi + 2, zi - 1)] +
                 data_[_index(xi - 1, yi + 2, zi - 1)] +
                 data_[_index(xi + 1, yi, zi - 1)] -
                 data_[_index(xi - 1, yi, zi - 1)]),
        0.125 *
            (data_[_index(xi + 2, yi + 2, zi + 1)] -
             data_[_index(xi, yi + 2, zi + 1)] -
             data_[_index(xi + 2, yi, zi + 1)] + data_[_index(xi, yi, zi + 1)] -
             data_[_index(xi + 2, yi + 2, zi - 1)] +
             data_[_index(xi, yi + 2, zi - 1)] +
             data_[_index(xi + 2, yi, zi - 1)] - data_[_index(xi, yi, zi - 1)]),
        0.125 * (data_[_index(xi + 1, yi + 1, zi + 2)] -
                 data_[_index(xi - 1, yi + 1, zi + 2)] -
                 data_[_index(xi + 1, yi - 1, zi + 2)] +
                 data_[_index(xi - 1, yi - 1, zi + 2)] -
                 data_[_index(xi + 1, yi + 1, zi)] +
                 data_[_index(xi - 1, yi + 1, zi)] +
                 data_[_index(xi + 1, yi - 1, zi)] -
                 data_[_index(xi - 1, yi - 1, zi)]),
        0.125 *
            (data_[_index(xi + 2, yi + 1, zi + 2)] -
             data_[_index(xi, yi + 1, zi + 2)] -
             data_[_index(xi + 2, yi - 1, zi + 2)] +
             data_[_index(xi, yi - 1, zi + 2)] -
             data_[_index(xi + 2, yi + 1, zi)] + data_[_index(xi, yi + 1, zi)] +
             data_[_index(xi + 2, yi - 1, zi)] - data_[_index(xi, yi - 1, zi)]),
        0.125 * (data_[_index(xi + 1, yi + 2, zi + 2)] -
                 data_[_index(xi - 1, yi + 2, zi + 2)] -
                 data_[_index(xi + 1, yi, zi + 2)] +
                 data_[_index(xi - 1, yi, zi + 2)] -
                 data_[_index(xi + 1, yi + 2, zi)] +
                 data_[_index(xi - 1, yi + 2, zi)] +
                 data_[_index(xi + 1, yi, zi)] - data_[_index(xi - 1, yi, zi)]),
        0.125 *
            (data_[_index(xi + 2, yi + 2, zi + 2)] -
             data_[_index(xi, yi + 2, zi + 2)] -
             data_[_index(xi + 2, yi, zi + 2)] + data_[_index(xi, yi, zi + 2)] -
             data_[_index(xi + 2, yi + 2, zi)] + data_[_index(xi, yi + 2, zi)] +
             data_[_index(xi + 2, yi, zi)] - data_[_index(xi, yi, zi)]);
    // Convert voxel values and partial derivatives
    // to interpolatio coefficients
    _coefs = _C * x;
    // Remember this voxel for next time.
    _i1 = xi;
    _i2 = yi;
    _i3 = zi;
    _initialized = true;
  }
  // Evaluate the interpolation within this grid voxel.
  dx -= xi;
  dy -= yi;
  dz -= zi;
  int ijkn(0);
  fptype dzpow(1);
  fptype result(0);
  for (int k = 0; k < 4; ++k) {
    fptype dypow(1);
    for (int j = 0; j < 4; ++j) {
      result += dypow * dzpow *
                (_coefs[ijkn] +
                 dx * (_coefs[ijkn + 1] +
                       dx * (_coefs[ijkn + 2] + dx * _coefs[ijkn + 3])));
      ijkn += 4;
      dypow *= dy;
    }
    dzpow *= dz;
  }
  return result;
}