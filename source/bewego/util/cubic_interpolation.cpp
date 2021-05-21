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

#include <bewego/util/cubic_interp_matrices.h>
#include <bewego/util/cubic_interpolation.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

using std::cout;
using std::endl;

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
    fptype x, fptype& dx) const {
  // determine the relative position in the
  // inteval enclosed by nearest data points
  dx = fmod(x / spacing_, n_);
  if (dx < 0) dx += n_;     // periodicity is built in
  int xi = (int)floor(dx);  // calculate lower-bound grid indices
  dx -= xi;                 // coordinate in interval

  Eigen::Matrix<fptype, 4, 1> p;
  p(0) = data_[xi - 1];
  p(1) = data_[xi];
  p(2) = data_[xi + 1];
  p(3) = data_[xi + 2];
  return p;
}

Eigen::Matrix<CubicInterpolator::fptype, 4, 1> CubicInterpolator::Coefficients(
    fptype x, fptype& dx) const {
  Eigen::Matrix<fptype, 4, 1> p = Neighboors(x, dx);
  return A_ * p;
}

CubicInterpolator::fptype CubicInterpolator::Evaluate(fptype x) const {
  double dx = 0;
  Eigen::Matrix<fptype, 4, 1> coef = Coefficients(x, dx);
  double dxpow2 = dx * dx;
  double dxpow3 = dxpow2 * dx;
  return coef(3) * dxpow3 + coef(2) * dxpow2 + coef(1) * dx + coef(0);
}

CubicInterpolator::fptype CubicInterpolator::Derivative(fptype x) const {
  double dx = 0;
  Eigen::Matrix<fptype, 4, 1> coef = Coefficients(x, dx);
  double dxpow2 = dx * dx;
  return 3. * coef(3) * dxpow2 + 2. * coef(2) * dx + coef(1);
}

double CubicInterpolator::Interpolate(const Eigen::Matrix<fptype, 4, 1>& p,
                                      fptype dx) {
  double dxpow2 = dx * dx;
  double dxpow3 = dxpow2 * dx;
  double val = (p(2) - p(0)) * dx +
               (2. * p(0) - 5. * p(1) + 4. * p(2) - p(3)) * dxpow2 +
               (-p(0) + 3. * p(1) - 3 * p(2) + p(3)) * dxpow3;
  return p(1) + 0.5 * val;
}

//------------------------------------------------------------------------------
// BiCubicGridInterpolator implementation
//------------------------------------------------------------------------------

BiCubicGridInterpolator::BiCubicGridInterpolator(
    const std::vector<fptype>& data, fptype spacing, int n1, int n2)
    : data_(data), n1_(n1), n2_(n2), spacing_(spacing) {
  if (n2_ == 0) {
    n2_ = n1_;
  }
  if (n1_ <= 0 || n2_ <= 0) {
    throw std::runtime_error("Bad datasquare dimensions.");
  }
  if (spacing_ <= 0) {
    throw std::runtime_error("Bad datasquare grid spacing.");
  }

  A_ = InitializeBiCubicMatrix();
}

Eigen::Matrix<CubicInterpolator::fptype, 16, 1>
BiCubicGridInterpolator::Neighboors(fptype x, fptype y, fptype& dx,
                                    fptype& dy) const {
  // determine the relative position in the
  // inteval enclosed by nearest data points
  dx = fmod(x / spacing_, n1_);
  dy = fmod(y / spacing_, n2_);

  if (dx < 0) dx += n1_;  // periodicity is built in
  if (dy < 0) dy += n2_;
  int xi = (int)floor(dx);  // calculate lower-bound grid indices
  int yi = (int)floor(dy);

  dx -= xi;
  dy -= yi;

  Eigen::Matrix<fptype, 16, 1> p;

  p(0) = data_[index_(xi - 1, yi - 1)];  // p00
  p(1) = data_[index_(xi + 0, yi - 1)];  // p10
  p(2) = data_[index_(xi + 1, yi - 1)];  // p20
  p(3) = data_[index_(xi + 2, yi - 1)];  // p30

  p(4) = data_[index_(xi - 1, yi - 0)];  // p01
  p(5) = data_[index_(xi + 0, yi - 0)];  // p11
  p(6) = data_[index_(xi + 1, yi - 0)];  // p21
  p(7) = data_[index_(xi + 2, yi - 0)];  // p31

  p(8) = data_[index_(xi - 1, yi + 1)];   // p02
  p(9) = data_[index_(xi + 0, yi + 1)];   // p12
  p(10) = data_[index_(xi + 1, yi + 1)];  // p22
  p(11) = data_[index_(xi + 2, yi + 1)];  // p32

  p(12) = data_[index_(xi - 1, yi + 2)];  // p03
  p(13) = data_[index_(xi + 0, yi + 2)];  // p13
  p(14) = data_[index_(xi + 1, yi + 2)];  // p23
  p(15) = data_[index_(xi + 2, yi + 2)];  // p33

  return p;
}

Eigen::Matrix<CubicInterpolator::fptype, 16, 1>
BiCubicGridInterpolator::Coefficients(fptype x, fptype y, fptype& dx,
                                      fptype& dy) const {
  auto p = Neighboors(x, y, dx, dy);
  Eigen::Matrix<fptype, 16, 1> a;

  // a00
  a(0) = p(1 + 4 * 1);
  // a01
  a(1) = -.5 * p(1 + 4 * 0) + .5 * p(1 + 4 * 2);
  // a02
  a(2) =
      p(1 + 4 * 0) - 2.5 * p(1 + 4 * 1) + 2 * p(1 + 4 * 2) - .5 * p(1 + 4 * 3);
  // a03
  a(3) = -.5 * p(1 + 4 * 0) + 1.5 * p(1 + 4 * 1) - 1.5 * p(1 + 4 * 2) +
         .5 * p(1 + 4 * 3);
  // a10
  a(4) = -.5 * p(0 + 4 * 1) + .5 * p(2 + 4 * 1);
  // a11
  a(5) = .25 * p(0 + 4 * 0) - .25 * p(0 + 4 * 2) - .25 * p(2 + 4 * 0) +
         .25 * p(2 + 4 * 2);
  // a12
  a(6) = -.5 * p(0 + 4 * 0) + 1.25 * p(0 + 4 * 1) - p(0 + 4 * 2) +
         .25 * p(0 + 4 * 3) + .5 * p(2 + 4 * 0) - 1.25 * p(2 + 4 * 1) +
         p(2 + 4 * 2) - .25 * p(2 + 4 * 3);
  // a13
  a(7) = .25 * p(0 + 4 * 0) - .75 * p(0 + 4 * 1) + .75 * p(0 + 4 * 2) -
         .25 * p(0 + 4 * 3) - .25 * p(2 + 4 * 0) + .75 * p(2 + 4 * 1) -
         .75 * p(2 + 4 * 2) + .25 * p(2 + 4 * 3);
  // a20
  a(8) =
      p(0 + 4 * 1) - 2.5 * p(1 + 4 * 1) + 2 * p(2 + 4 * 1) - .5 * p(3 + 4 * 1);
  // a21
  a(9) = -.5 * p(0 + 4 * 0) + .5 * p(0 + 4 * 2) + 1.25 * p(1 + 4 * 0) -
         1.25 * p(1 + 4 * 2) - p(2 + 4 * 0) + p(2 + 4 * 2) +
         .25 * p(3 + 4 * 0) - .25 * p(3 + 4 * 2);
  // a22
  a(10) = p(0 + 4 * 0) - 2.5 * p(0 + 4 * 1) + 2 * p(0 + 4 * 2) -
          .5 * p(0 + 4 * 3) - 2.5 * p(1 + 4 * 0) + 6.25 * p(1 + 4 * 1) -
          5 * p(1 + 4 * 2) + 1.25 * p(1 + 4 * 3) + 2 * p(2 + 4 * 0) -
          5 * p(2 + 4 * 1) + 4 * p(2 + 4 * 2) - p(2 + 4 * 3) -
          .5 * p(3 + 4 * 0) + 1.25 * p(3 + 4 * 1) - p(3 + 4 * 2) +
          .25 * p(3 + 4 * 3);
  // a23
  a(11) = -.5 * p(0 + 4 * 0) + 1.5 * p(0 + 4 * 1) - 1.5 * p(0 + 4 * 2) +
          .5 * p(0 + 4 * 3) + 1.25 * p(1 + 4 * 0) - 3.75 * p(1 + 4 * 1) +
          3.75 * p(1 + 4 * 2) - 1.25 * p(1 + 4 * 3) - p(2 + 4 * 0) +
          3 * p(2 + 4 * 1) - 3 * p(2 + 4 * 2) + p(2 + 4 * 3) +
          .25 * p(3 + 4 * 0) - .75 * p(3 + 4 * 1) + .75 * p(3 + 4 * 2) -
          .25 * p(3 + 4 * 3);
  // a30
  a(12) = -.5 * p(0 + 4 * 1) + 1.5 * p(1 + 4 * 1) - 1.5 * p(2 + 4 * 1) +
          .5 * p(3 + 4 * 1);
  // a31
  a(13) = .25 * p(0 + 4 * 0) - .25 * p(0 + 4 * 2) - .75 * p(1 + 4 * 0) +
          .75 * p(1 + 4 * 2) + .75 * p(2 + 4 * 0) - .75 * p(2 + 4 * 2) -
          .25 * p(3 + 4 * 0) + .25 * p(3 + 4 * 2);
  // a32
  a(14) = -.5 * p(0 + 4 * 0) + 1.25 * p(0 + 4 * 1) - p(0 + 4 * 2) +
          .25 * p(0 + 4 * 3) + 1.5 * p(1 + 4 * 0) - 3.75 * p(1 + 4 * 1) +
          3 * p(1 + 4 * 2) - .75 * p(1 + 4 * 3) - 1.5 * p(2 + 4 * 0) +
          3.75 * p(2 + 4 * 1) - 3 * p(2 + 4 * 2) + .75 * p(2 + 4 * 3) +
          .5 * p(3 + 4 * 0) - 1.25 * p(3 + 4 * 1) + p(3 + 4 * 2) -
          .25 * p(3 + 4 * 3);
  // a33
  a(15) = .25 * p(0 + 4 * 0) - .75 * p(0 + 4 * 1) + .75 * p(0 + 4 * 2) -
          .25 * p(0 + 4 * 3) - .75 * p(1 + 4 * 0) + 2.25 * p(1 + 4 * 1) -
          2.25 * p(1 + 4 * 2) + .75 * p(1 + 4 * 3) + .75 * p(2 + 4 * 0) -
          2.25 * p(2 + 4 * 1) + 2.25 * p(2 + 4 * 2) - .75 * p(2 + 4 * 3) -
          .25 * p(3 + 4 * 0) + .75 * p(3 + 4 * 1) - .75 * p(3 + 4 * 2) +
          .25 * p(3 + 4 * 3);
  return a;
}

BiCubicGridInterpolator::fptype BiCubicGridInterpolator::Evaluate(
    const Eigen::Matrix<BiCubicGridInterpolator::fptype, 2, 1>& point) const {
  double dx, dy;
  auto a = Coefficients(point.x(), point.y(), dx, dy);

  double dxpow2 = dx * dx;
  double dxpow3 = dxpow2 * dx;
  double dypow2 = dy * dy;
  double dypow3 = dypow2 * dy;

  return (a(0) + a(1) * dy + a(2) * dypow2 + a(3) * dypow3) +
         (a(4) + a(5) * dy + a(6) * dypow2 + a(7) * dypow3) * dx +
         (a(8) + a(9) * dy + a(10) * dypow2 + a(11) * dypow3) * dxpow2 +
         (a(12) + a(13) * dy + a(14) * dypow2 + a(15) * dypow3) * dxpow3;
}

Eigen::Matrix<BiCubicGridInterpolator::fptype, 2, 1>
BiCubicGridInterpolator::Gradient(
    const Eigen::Matrix<BiCubicGridInterpolator::fptype, 2, 1>& point) const {
  double dx, dy;
  auto a = Coefficients(point.x(), point.y(), dx, dy);

  double dxpow2 = dx * dx;
  double dxpow3 = dxpow2 * dx;
  double dypow2 = dy * dy;
  double dypow3 = dypow2 * dy;

  double dvx =
      // Constant
      a(4) + a(5) * dy + a(6) * dypow2 + a(7) * dypow3 +
      // 1st order
      2 * (a(8) + a(9) * dy + a(10) * dypow2 + a(11) * dypow3) * dx +
      // 2st order
      3 * (a(12) + a(13) * dy + a(14) * dypow2 + a(15) * dypow3) * dxpow2;

  double dvy =
      // Constant
      a(1) + a(5) * dx + a(9) * dxpow2 + a(13) * dxpow3 +
      // 1st order
      2. * (a(2) + a(6) * dx + a(10) * dxpow2 + a(14) * dxpow3) * dy +
      // 2st order
      3. * (a(3) + a(7) * dx + a(11) * dxpow2 + a(15) * dxpow3) * dypow2;

  return Eigen::Matrix<fptype, 2, 1>(dvx, dvy);
}

BiCubicGridInterpolator::~BiCubicGridInterpolator() {}

BiCubicGridInterpolator::fptype BiCubicGridInterpolator::Interpolate(
    const Eigen::Matrix<BiCubicGridInterpolator::fptype, 16, 1>& p, fptype dx,
    fptype dy) {
  Eigen::Matrix<fptype, 4, 1> arr;
  typedef const Eigen::Matrix<fptype, 4, 4, Eigen::RowMajor> MatrixType;
  MatrixType neigh = Eigen::Map<MatrixType>(p.data());
  arr[0] = CubicInterpolator::Interpolate(neigh.row(0), dy);
  arr[1] = CubicInterpolator::Interpolate(neigh.row(1), dy);
  arr[2] = CubicInterpolator::Interpolate(neigh.row(2), dy);
  arr[3] = CubicInterpolator::Interpolate(neigh.row(3), dy);
  return CubicInterpolator::Interpolate(arr, dx);
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

  _C1 = InitializeC1Matrix();
  _C2 = InitializeC2Matrix();
}

TriCubicGridInterpolator::~TriCubicGridInterpolator() {}

TriCubicGridInterpolator::fptype TriCubicGridInterpolator::Evaluate(
    const Eigen::Matrix<fptype, 3, 1>& point) {
  fptype x = point.x();
  fptype y = point.y();
  fptype z = point.z();

  // determine the relative position in the
  // box enclosed by nearest data points
  fptype dx = fmod(x / _spacing, _n1);
  fptype dy = fmod(y / _spacing, _n2);
  fptype dz = fmod(z / _spacing, _n3);

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
