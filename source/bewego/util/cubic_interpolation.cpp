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

#include <bewego/util/cubic_interpolation.h>

#include <array>
#include <stdexcept>
#include <string>

//------------------------------------------------------------------------------
// CubicInterpolator implementation
//------------------------------------------------------------------------------

CubicInterpolator::CubicInterpolator(const std::vector<fptype>& data,
                                     fptype spacing)
    : data_(data), n_(data.size()), spacing_(spacing) {
  A_.row(0) << 0, 1, 0, 0;
  A_.row(1) << -.5, 0, .5, 0;
  A_.row(2) << 1, -2.5, 2, -.5;
  A_.row(3) << -.5, 1.5, -1.5, .5;
}

CubicInterpolator::~CubicInterpolator() {}

Eigen::Matrix<CubicInterpolator::fptype, 4, 1> CubicInterpolator::Neighboors(
    CubicInterpolator::fptype x) const {
  // determine the relative position in the
  // inteval enclosed by nearest data points
  fptype dx = fmod(x / spacing_, n_);

  if (dx < 0) dx += n_;     // periodicity is built in
  int xi = (int)floor(dx);  // calculate lower-bound grid indices

  Eigen::Matrix<fptype, 4, 1> p;
  p(0) = data_[xi - 1];
  p(1) = data_[xi];
  p(2) = data_[xi + 1];
  p(3) = data_[xi + 2];

  return p;
}

Eigen::Matrix<CubicInterpolator::fptype, 4, 1> CubicInterpolator::Coefficients(
    CubicInterpolator::fptype x) const {
  Eigen::Matrix<fptype, 4, 1> p = Neighboors(x);
  return A_ * p;
}

CubicInterpolator::fptype CubicInterpolator::Evaluate(
    CubicInterpolator::fptype x) const {
  Eigen::Matrix<fptype, 4, 1> coef = Coefficients(x);
  double xpow2 = x * x;
  double xpow3 = xpow2 * x;
  return coef(3) * xpow3 + coef(2) * xpow2 + coef(1) * x + coef(0);
}

CubicInterpolator::fptype CubicInterpolator::Derivative(fptype x) const {
  Eigen::Matrix<fptype, 4, 1> coef = Coefficients(x);
  double xpow2 = x * x;
  double xpow3 = xpow2 * x;
  return 3. * coef(3) * xpow2 + 2. * coef(2) * x + coef(1);
}

double CubicInterpolator::Interpolate(const Eigen::Matrix<fptype, 4, 1>& p,
                                      fptype x) {
  double xpow2 = x * x;
  double xpow3 = xpow2 * x;
  double val = (p(2) - p(0)) * x +
               (2. * p(0) - 5. * p(1) + 4. * p(2) - p(3)) * xpow2 +
               (-p(0) + 3. * p(1) - 3 * p(2) + p(3)) * xpow3;
  return p(1) + 0.5 * val;
}

//------------------------------------------------------------------------------
// BiCubicGridInterpolator implementation
//------------------------------------------------------------------------------

BiCubicGridInterpolator::BiCubicGridInterpolator(
    const std::vector<fptype>& data, fptype spacing, int n1, int n2) {}

BiCubicGridInterpolator::fptype BiCubicGridInterpolator::Evaluate(
    const Eigen::Matrix<fptype, 2, 1>& point) const {
  // TODO
  return 0;
}

BiCubicGridInterpolator::~BiCubicGridInterpolator() {}

BiCubicGridInterpolator::fptype BiCubicGridInterpolator::Interpolate(
    BiCubicGridInterpolator::fptype p[4][4], double x,
    BiCubicGridInterpolator::fptype y) {
  // double arr[4];
  // arr[0] = CubicInterpolator::Interpolate(p[0], y);
  // arr[1] = CubicInterpolator::Interpolate(p[1], y);
  // arr[2] = CubicInterpolator::Interpolate(p[2], y);
  // arr[3] = CubicInterpolator::Interpolate(p[3], y);
  // return CubicInterpolator::Interpolate(arr, x);
  return 0;
}

//------------------------------------------------------------------------------
// TriCubicGridInterpolator implementation
//------------------------------------------------------------------------------

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

  InitializeC1Matrix();
  InitializeC2Matrix();
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
    auto x = FiniteDifferenceDataAtCorners(xi, yi, zi);
    // Convert voxel values and partial derivatives
    // to interpolation coefficients
    _coefs = _C1 * x;
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

Eigen::Matrix<TriCubicGridInterpolator::fptype, 64, 1>
TriCubicGridInterpolator::FiniteDifferenceDataAtCorners(int xi, int yi,
                                                        int zi) const {
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
      0.5 * (data_[_index(xi + 2, yi + 1, zi)] - data_[_index(xi, yi + 1, zi)]),
      0.5 * (data_[_index(xi + 1, yi, zi + 1)] -
             data_[_index(xi - 1, yi, zi + 1)]),
      0.5 * (data_[_index(xi + 2, yi, zi + 1)] - data_[_index(xi, yi, zi + 1)]),
      0.5 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
             data_[_index(xi - 1, yi + 1, zi + 1)]),
      0.5 * (data_[_index(xi + 2, yi + 1, zi + 1)] -
             data_[_index(xi, yi + 1, zi + 1)]),
      // values of df/dy at each corner.
      0.5 * (data_[_index(xi, yi + 1, zi)] - data_[_index(xi, yi - 1, zi)]),
      0.5 * (data_[_index(xi + 1, yi + 1, zi)] -
             data_[_index(xi + 1, yi - 1, zi)]),
      0.5 * (data_[_index(xi, yi + 2, zi)] - data_[_index(xi, yi, zi)]),
      0.5 * (data_[_index(xi + 1, yi + 2, zi)] - data_[_index(xi + 1, yi, zi)]),
      0.5 * (data_[_index(xi, yi + 1, zi + 1)] -
             data_[_index(xi, yi - 1, zi + 1)]),
      0.5 * (data_[_index(xi + 1, yi + 1, zi + 1)] -
             data_[_index(xi + 1, yi - 1, zi + 1)]),
      0.5 * (data_[_index(xi, yi + 2, zi + 1)] - data_[_index(xi, yi, zi + 1)]),
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
      0.5 * (data_[_index(xi + 1, yi, zi + 2)] - data_[_index(xi + 1, yi, zi)]),
      0.5 * (data_[_index(xi, yi + 1, zi + 2)] - data_[_index(xi, yi + 1, zi)]),
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
  return x;
}

void TriCubicGridInterpolator::InitializeC1Matrix() {
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
    for (int j = 0; j < 64; j++) _C1(i, j) = temp[i][j];
}

void TriCubicGridInterpolator::InitializeC2Matrix() {
  /*
  The whole idea of tri-cubic interpolation
  is the setup a linear system of equation which has a unique
  solution. In such a case it the only thing to do is to invert
  the matrix A (called C1 in the code) and multiply the data b of derivatives at
  the corner to obtain the tricubic polynomial coefficients x within the
  element:

        x = A^{-1}b .

  Typically, the derivatives at the corner of the element are obtained using
  finite differences, which can be packed in a matrix such that

        b = By

  where B is the finite difference matrix and y is the data in a 4x4x4
  cube arround the element.

  The following matrix C2 implements:

    C2 = A^{-1} B

  which should allow a significant speedum when computing the
  spline coefficents.

  See this page

   TriCubic splines can be accelerated using (see)
   http://ianfaust.com/2016/03/20/Tricubic/

  Code can be found
   https://github.com/PSFCPlasmaTools/eqtools/blob/master/eqtools/_tricub.c
   */
  const int temp[64][64] = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0,
       4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, -20,
       16, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 12,
       -12, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,
       0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0,
       0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0},
      {0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, 0, 0,
       0, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,
       0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0},
      {0, 0, 0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 10, -8, 2, 0, 0,
       0, 0, 4, -10, 8, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0,
       0, 0, 0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0},
      {0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -6, 6, -2, 0, 0,
       0, 0, -2, 6, -6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0,
       0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0},
      {0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, -20,
       0, 0, 0, 16, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0,   0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, -4, 0, 4, 0, 10, 0,
       -10, 0, -8, 0, 8, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,  0,
       0,   0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0},
      {0, 0,   0,  0,  0,   0,  0,   0,  0,  0,   0,  0,  0,  0,  0,  0,
       8, -20, 16, -4, -20, 50, -40, 10, 16, -40, 32, -8, -4, 10, -8, 2,
       0, 0,   0,  0,  0,   0,  0,   0,  0,  0,   0,  0,  0,  0,  0,  0,
       0, 0,   0,  0,  0,   0,  0,   0,  0,  0,   0,  0,  0,  0,  0,  0},
      {0,  0,  0,   0, 0,  0,   0,  0,   0,  0,  0,   0, 0, 0,  0, 0,
       -4, 12, -12, 4, 10, -30, 30, -10, -8, 24, -24, 8, 2, -6, 6, -2,
       0,  0,  0,   0, 0,  0,   0,  0,   0,  0,  0,   0, 0, 0,  0, 0,
       0,  0,  0,   0, 0,  0,   0,  0,   0,  0,  0,   0, 0, 0,  0, 0},
      {0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 12,
       0, 0, 0, -12, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,
       0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0},
      {0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2, 0, -6, 0,
       6, 0, 6, 0, -6, 0, -2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0,
       0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0},
      {0,  0,  0,  0, 0,  0,   0,  0,  0,   0,  0,   0, 0, 0,   0, 0,
       -4, 10, -8, 2, 12, -30, 24, -6, -12, 30, -24, 6, 4, -10, 8, -2,
       0,  0,  0,  0, 0,  0,   0,  0,  0,   0,  0,   0, 0, 0,   0, 0,
       0,  0,  0,  0, 0,  0,   0,  0,  0,   0,  0,   0, 0, 0,   0, 0},
      {0, 0,  0, 0,  0,  0,  0,   0, 0, 0,   0,  0,  0,  0, 0,  0,
       2, -6, 6, -2, -6, 18, -18, 6, 6, -18, 18, -6, -2, 6, -6, 2,
       0, 0,  0, 0,  0,  0,  0,   0, 0, 0,   0,  0,  0,  0, 0,  0,
       0, 0,  0, 0,  0,  0,  0,   0, 0, 0,   0,  0,  0,  0, 0,  0},
      {0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, -2, 0, 2, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0},
      {0, 0, 0, 0, -4, 10, -8, 2, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,  0, 0, 0, 0,
       0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 4, -10, 8, -2, 0, 0, 0, 0,
       0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,  0, 0},
      {0, 0, 0, 0, 2, -6, 6, -2, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, -2, 6, -6, 2, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0},
      {0, 2, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, -2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0},
      {-1, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0, -1, 0, 0, 0, 0, 0, -1, 0, 1, 0,
       0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,  0},
      {2, -5, 4, -1, 0, 0, 0, 0, -2, 5, -4, 1, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,
       0, 0,  0, 0,  0, 0, 0, 0, 0,  0, -2, 5, -4, 1, 0, 0, 0, 0, 2, -5, 4, -1,
       0, 0,  0, 0,  0, 0, 0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0},
      {-1, 3, -3, 1, 0, 0, 0, 0, 1, -3, 3, -1, 0, 0,  0, 0, 0, 0, 0,  0, 0,  0,
       0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  1, -3, 3, -1, 0, 0, 0, 0, -1, 3, -3, 1,
       0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0,  0, 0, 0, 0, 0,  0},
      {0, -4, 0, 0, 0, 10, 0, 0, 0, -8, 0, 0, 0, 2, 0, 0,   0, 0, 0, 0, 0, 0,
       0, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 4, 0, 0, 0, -10, 0, 0, 0, 8, 0, 0,
       0, -2, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0,   0, 0, 0, 0},
      {2, 0, -2, 0, -5, 0, 5, 0, 4, 0, -4, 0, -1, 0, 1, 0, 0,  0, 0,  0, 0, 0,
       0, 0, 0,  0, 0,  0, 0, 0, 0, 0, -2, 0, 2,  0, 5, 0, -5, 0, -4, 0, 4, 0,
       1, 0, -1, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0,  0, 0,  0},
      {-4, 10,  -8, 2,  10,  -25, 20,  -5, -8, 20,  -16, 4,  2,  -5, 4,  -1,
       0,  0,   0,  0,  0,   0,   0,   0,  0,  0,   0,   0,  0,  0,  0,  0,
       4,  -10, 8,  -2, -10, 25,  -20, 5,  8,  -20, 16,  -4, -2, 5,  -4, 1,
       0,  0,   0,  0,  0,   0,   0,   0,  0,  0,   0,   0,  0,  0,  0,  0},
      {2,  -6, 6,  -2, -5, 15,  -15, 5,  4,  -12, 12,  -4, -1, 3,  -3, 1,
       0,  0,  0,  0,  0,  0,   0,   0,  0,  0,   0,   0,  0,  0,  0,  0,
       -2, 6,  -6, 2,  5,  -15, 15,  -5, -4, 12,  -12, 4,  1,  -3, 3,  -1,
       0,  0,  0,  0,  0,  0,   0,   0,  0,  0,   0,   0,  0,  0,  0,  0},
      {0, 2, 0, 0, 0, -6, 0, 0, 0, 6, 0, 0,  0, -2, 0, 0, 0, 0, 0, 0,  0, 0,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, -2, 0, 0,  0, 6, 0, 0, 0, -6, 0, 0,
       0, 2, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0},
      {-1, 0, 1, 0, 3, 0, -3, 0, -3, 0, 3, 0, 1,  0, -1, 0, 0, 0, 0, 0, 0,  0,
       0,  0, 0, 0, 0, 0, 0,  0, 0,  0, 1, 0, -1, 0, -3, 0, 3, 0, 3, 0, -3, 0,
       -1, 0, 1, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0},
      {2,  -5, 4,  -1, -6, 15,  -12, 3,  6,  -15, 12,  -3, -2, 5,  -4, 1,
       0,  0,  0,  0,  0,  0,   0,   0,  0,  0,   0,   0,  0,  0,  0,  0,
       -2, 5,  -4, 1,  6,  -15, 12,  -3, -6, 15,  -12, 3,  2,  -5, 4,  -1,
       0,  0,  0,  0,  0,  0,   0,   0,  0,  0,   0,   0,  0,  0,  0,  0},
      {-1, 3,  -3, 1,  3,  -9, 9,  -3, -3, 9,  -9, 3,  1,  -3, 3,  -1,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       1,  -3, 3,  -1, -3, 9,  -9, 3,  3,  -9, 9,  -3, -1, 3,  -3, 1,
       0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {0, 0, 0, 0, 0, 8, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, -20,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0},
      {0,   0, 0, 0, -4, 0, 4, 0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 10, 0,
       -10, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,  0, 0, 0, -8, 0, 8, 0, 0, 0, 0,  0,
       0,   0, 0, 0, 0,  0, 0, 0, 2, 0, -2, 0, 0, 0, 0,  0, 0, 0, 0, 0},
      {0, 0, 0, 0, 8,   -20, 16,  -4, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -20, 50,  -40, 10, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 16,  -40, 32,  -8, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -4,  10,  -8,  2,  0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, -4, 12,  -12, 4,   0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 10, -30, 30,  -10, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -8, 24,  -24, 8,   0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2,  -6,  6,   -2,  0, 0, 0, 0, 0, 0, 0, 0},
      {0, -4, 0, 0,   0, 0, 0, 0, 0, 4, 0, 0,  0, 0,  0, 0, 0, 10, 0, 0, 0, 0,
       0, 0,  0, -10, 0, 0, 0, 0, 0, 0, 0, -8, 0, 0,  0, 0, 0, 0,  0, 8, 0, 0,
       0, 0,  0, 0,   0, 2, 0, 0, 0, 0, 0, 0,  0, -2, 0, 0, 0, 0,  0, 0},
      {2, 0, -2, 0, 0,  0, 0, 0, -2, 0, 2, 0, 0,  0, 0,  0, -5, 0, 5,  0, 0, 0,
       0, 0, 5,  0, -5, 0, 0, 0, 0,  0, 4, 0, -4, 0, 0,  0, 0,  0, -4, 0, 4, 0,
       0, 0, 0,  0, -1, 0, 1, 0, 0,  0, 0, 0, 1,  0, -1, 0, 0,  0, 0,  0},
      {-4, 10,  -8,  2,  0, 0, 0, 0, 4,   -10, 8,   -2, 0, 0, 0, 0,
       10, -25, 20,  -5, 0, 0, 0, 0, -10, 25,  -20, 5,  0, 0, 0, 0,
       -8, 20,  -16, 4,  0, 0, 0, 0, 8,   -20, 16,  -4, 0, 0, 0, 0,
       2,  -5,  4,   -1, 0, 0, 0, 0, -2,  5,   -4,  1,  0, 0, 0, 0},
      {2,  -6,  6,   -2, 0, 0, 0, 0, -2, 6,   -6,  2,  0, 0, 0, 0,
       -5, 15,  -15, 5,  0, 0, 0, 0, 5,  -15, 15,  -5, 0, 0, 0, 0,
       4,  -12, 12,  -4, 0, 0, 0, 0, -4, 12,  -12, 4,  0, 0, 0, 0,
       -1, 3,   -3,  1,  0, 0, 0, 0, 1,  -3,  3,   -1, 0, 0, 0, 0},
      {0, 8,   0, 0, 0, -20, 0, 0, 0, 16,  0, 0, 0, -4, 0, 0,
       0, -20, 0, 0, 0, 50,  0, 0, 0, -40, 0, 0, 0, 10, 0, 0,
       0, 16,  0, 0, 0, -40, 0, 0, 0, 32,  0, 0, 0, -8, 0, 0,
       0, -4,  0, 0, 0, 10,  0, 0, 0, -8,  0, 0, 0, 2,  0, 0},
      {-4, 0, 4,   0, 10,  0, -10, 0, -8,  0, 8,   0, 2,  0, -2, 0,
       10, 0, -10, 0, -25, 0, 25,  0, 20,  0, -20, 0, -5, 0, 5,  0,
       -8, 0, 8,   0, 20,  0, -20, 0, -16, 0, 16,  0, 4,  0, -4, 0,
       2,  0, -2,  0, -5,  0, 5,   0, 4,   0, -4,  0, -1, 0, 1,  0},
      {8,   -20, 16,  -4,  -20, 50,  -40, 10,  16,   -40, 32,  -8,  -4,
       10,  -8,  2,   -20, 50,  -40, 10,  50,  -125, 100, -25, -40, 100,
       -80, 20,  10,  -25, 20,  -5,  16,  -40, 32,   -8,  -40, 100, -80,
       20,  32,  -80, 64,  -16, -8,  20,  -16, 4,    -4,  10,  -8,  2,
       10,  -25, 20,  -5,  -8,  20,  -16, 4,   2,    -5,  4,   -1},
      {-4,  12,  -12, 4,   10,  -30, 30,  -10, -8,  24,  -24, 8,   2,
       -6,  6,   -2,  10,  -30, 30,  -10, -25, 75,  -75, 25,  20,  -60,
       60,  -20, -5,  15,  -15, 5,   -8,  24,  -24, 8,   20,  -60, 60,
       -20, -16, 48,  -48, 16,  4,   -12, 12,  -4,  2,   -6,  6,   -2,
       -5,  15,  -15, 5,   4,   -12, 12,  -4,  -1,  3,   -3,  1},
      {0, -4, 0, 0, 0, 12,  0, 0, 0, -12, 0, 0, 0, 4,   0, 0,
       0, 10, 0, 0, 0, -30, 0, 0, 0, 30,  0, 0, 0, -10, 0, 0,
       0, -8, 0, 0, 0, 24,  0, 0, 0, -24, 0, 0, 0, 8,   0, 0,
       0, 2,  0, 0, 0, -6,  0, 0, 0, 6,   0, 0, 0, -2,  0, 0},
      {2,  0, -2, 0, -6,  0, 6,   0, 6,   0, -6,  0, -2, 0, 2,  0,
       -5, 0, 5,  0, 15,  0, -15, 0, -15, 0, 15,  0, 5,  0, -5, 0,
       4,  0, -4, 0, -12, 0, 12,  0, 12,  0, -12, 0, -4, 0, 4,  0,
       -1, 0, 1,  0, 3,   0, -3,  0, -3,  0, 3,   0, 1,  0, -1, 0},
      {-4,  10,  -8,  2,   12,  -30, 24,  -6,  -12, 30,  -24, 6,   4,
       -10, 8,   -2,  10,  -25, 20,  -5,  -30, 75,  -60, 15,  30,  -75,
       60,  -15, -10, 25,  -20, 5,   -8,  20,  -16, 4,   24,  -60, 48,
       -12, -24, 60,  -48, 12,  8,   -20, 16,  -4,  2,   -5,  4,   -1,
       -6,  15,  -12, 3,   6,   -15, 12,  -3,  -2,  5,   -4,  1},
      {2,   -6, 6,   -2,  -6,  18,  -18, 6,   6,   -18, 18,  -6,  -2,
       6,   -6, 2,   -5,  15,  -15, 5,   15,  -45, 45,  -15, -15, 45,
       -45, 15, 5,   -15, 15,  -5,  4,   -12, 12,  -4,  -12, 36,  -36,
       12,  12, -36, 36,  -12, -4,  12,  -12, 4,   -1,  3,   -3,  1,
       3,   -9, 9,   -3,  -3,  9,   -9,  3,   1,   -3,  3,   -1},
      {0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 12,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, -12, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0,  0, 0, 0, 4, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0},
      {0, 0, 0, 0, 2, 0, -2, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, -6, 0,
       6, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 6, 0, -6, 0, 0, 0, 0,  0,
       0, 0, 0, 0, 0, 0, 0,  0, -2, 0, 2, 0, 0, 0, 0, 0, 0,  0, 0, 0},
      {0, 0, 0, 0, -4,  10,  -8,  2,  0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 12,  -30, 24,  -6, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -12, 30,  -24, 6,  0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 4,   -10, 8,   -2, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 2,  -6,  6,   -2, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -6, 18,  -18, 6,  0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 6,  -18, 18,  -6, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -2, 6,   -6,  2,  0, 0, 0, 0, 0, 0, 0, 0},
      {0, 2, 0, 0, 0, 0,  0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0,  0, 0,
       0, 0, 0, 6, 0, 0,  0, 0, 0, 0,  0, 6, 0, 0, 0, 0, 0, 0,  0, -6, 0, 0,
       0, 0, 0, 0, 0, -2, 0, 0, 0, 0,  0, 0, 0, 2, 0, 0, 0, 0,  0, 0},
      {-1, 0, 1,  0, 0, 0, 0,  0, 1, 0, -1, 0, 0,  0, 0, 0, 3, 0, -3, 0, 0,  0,
       0,  0, -3, 0, 3, 0, 0,  0, 0, 0, -3, 0, 3,  0, 0, 0, 0, 0, 3,  0, -3, 0,
       0,  0, 0,  0, 1, 0, -1, 0, 0, 0, 0,  0, -1, 0, 1, 0, 0, 0, 0,  0},
      {2,  -5,  4,   -1, 0, 0, 0, 0, -2, 5,   -4,  1,  0, 0, 0, 0,
       -6, 15,  -12, 3,  0, 0, 0, 0, 6,  -15, 12,  -3, 0, 0, 0, 0,
       6,  -15, 12,  -3, 0, 0, 0, 0, -6, 15,  -12, 3,  0, 0, 0, 0,
       -2, 5,   -4,  1,  0, 0, 0, 0, 2,  -5,  4,   -1, 0, 0, 0, 0},
      {-1, 3,  -3, 1,  0, 0, 0, 0, 1,  -3, 3,  -1, 0, 0, 0, 0,
       3,  -9, 9,  -3, 0, 0, 0, 0, -3, 9,  -9, 3,  0, 0, 0, 0,
       -3, 9,  -9, 3,  0, 0, 0, 0, 3,  -9, 9,  -3, 0, 0, 0, 0,
       1,  -3, 3,  -1, 0, 0, 0, 0, -1, 3,  -3, 1,  0, 0, 0, 0},
      {0, -4,  0, 0, 0, 10,  0, 0, 0, -8,  0, 0, 0, 2,  0, 0,
       0, 12,  0, 0, 0, -30, 0, 0, 0, 24,  0, 0, 0, -6, 0, 0,
       0, -12, 0, 0, 0, 30,  0, 0, 0, -24, 0, 0, 0, 6,  0, 0,
       0, 4,   0, 0, 0, -10, 0, 0, 0, 8,   0, 0, 0, -2, 0, 0},
      {2,  0, -2, 0, -5,  0, 5,   0, 4,   0, -4,  0, -1, 0, 1,  0,
       -6, 0, 6,  0, 15,  0, -15, 0, -12, 0, 12,  0, 3,  0, -3, 0,
       6,  0, -6, 0, -15, 0, 15,  0, 12,  0, -12, 0, -3, 0, 3,  0,
       -2, 0, 2,  0, 5,   0, -5,  0, -4,  0, 4,   0, 1,  0, -1, 0},
      {-4,  10,  -8,  2,   10,  -25, 20,  -5,  -8,  20,  -16, 4,   2,
       -5,  4,   -1,  12,  -30, 24,  -6,  -30, 75,  -60, 15,  24,  -60,
       48,  -12, -6,  15,  -12, 3,   -12, 30,  -24, 6,   30,  -75, 60,
       -15, -24, 60,  -48, 12,  6,   -15, 12,  -3,  4,   -10, 8,   -2,
       -10, 25,  -20, 5,   8,   -20, 16,  -4,  -2,  5,   -4,  1},
      {2,  -6,  6,   -2, -5,  15, -15, 5,   4,  -12, 12, -4, -1, 3, -3, 1, -6,
       18, -18, 6,   15, -45, 45, -15, -12, 36, -36, 12, 3,  -9, 9, -3, 6, -18,
       18, -6,  -15, 45, -45, 15, 12,  -36, 36, -12, -3, 9,  -9, 3, -2, 6, -6,
       2,  5,   -15, 15, -5,  -4, 12,  -12, 4,  1,   -3, 3,  -1},
      {0, 2,  0, 0, 0, -6,  0, 0, 0, 6,   0, 0, 0, -2, 0, 0,
       0, -6, 0, 0, 0, 18,  0, 0, 0, -18, 0, 0, 0, 6,  0, 0,
       0, 6,  0, 0, 0, -18, 0, 0, 0, 18,  0, 0, 0, -6, 0, 0,
       0, -2, 0, 0, 0, 6,   0, 0, 0, -6,  0, 0, 0, 2,  0, 0},
      {-1, 0, 1,  0, 3,  0, -3, 0, -3, 0, 3,  0, 1,  0, -1, 0,
       3,  0, -3, 0, -9, 0, 9,  0, 9,  0, -9, 0, -3, 0, 3,  0,
       -3, 0, 3,  0, 9,  0, -9, 0, -9, 0, 9,  0, 3,  0, -3, 0,
       1,  0, -1, 0, -3, 0, 3,  0, 3,  0, -3, 0, -1, 0, 1,  0},
      {2,   -5,  4,   -1,  -6, 15,  -12, 3,   6,   -15, 12,  -3,  -2,
       5,   -4,  1,   -6,  15, -12, 3,   18,  -45, 36,  -9,  -18, 45,
       -36, 9,   6,   -15, 12, -3,  6,   -15, 12,  -3,  -18, 45,  -36,
       9,   18,  -45, 36,  -9, -6,  15,  -12, 3,   -2,  5,   -4,  1,
       6,   -15, 12,  -3,  -6, 15,  -12, 3,   2,   -5,  4,   -1},
      {-1, 3,  -3, 1,  3,  -9,  9,   -3, -3, 9,   -9,  3,  1,  -3, 3,  -1,
       3,  -9, 9,  -3, -9, 27,  -27, 9,  9,  -27, 27,  -9, -3, 9,  -9, 3,
       -3, 9,  -9, 3,  9,  -27, 27,  -9, -9, 27,  -27, 9,  3,  -9, 9,  -3,
       1,  -3, 3,  -1, -3, 9,   -9,  3,  3,  -9,  9,   -3, -1, 3,  -3, 1}};

  for (int i = 0; i < 64; i++)
    for (int j = 0; j < 64; j++) _C2(i, j) = temp[i][j];
}