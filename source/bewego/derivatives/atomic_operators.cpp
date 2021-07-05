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
 *                                                             Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/derivatives/atomic_operators.h>

using namespace bewego;
using std::cout;
using std::endl;

//------------------------------------------------------------------------------
// LinearMap implementation
//------------------------------------------------------------------------------

Eigen::VectorXd LinearMap::Forward(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  return A_ * x;
}

//------------------------------------------------------------------------------
// SecondOrderTaylorApproximation implementation
//------------------------------------------------------------------------------

SecondOrderTaylorApproximation::SecondOrderTaylorApproximation(
    const DifferentiableMap& f, const Eigen::VectorXd& x0)
    : x0_(x0) {
  assert(f.output_dimension() == 1);
  double v = f.ForwardFunc(x0);
  Eigen::VectorXd g = f.Gradient(x0);
  Eigen::MatrixXd H = f.Hessian(x0);
  double c = 0;

  H_ = H;
  a_ = H;
  b_ = g - x0.transpose() * H;
  c = v - g.transpose() * x0 + .5 * x0.transpose() * H * x0;
  c_ = Eigen::VectorXd::Constant(1, c);

  x0_ = x0;
  g0_ = g;
  fx0_ = v;

  type_ = "SecondOrderTaylorApproximation";
}

//------------------------------------------------------------------------------
// LogBarrier implementation
//------------------------------------------------------------------------------

Eigen::VectorXd LogBarrier::Forward(const Eigen::VectorXd& x_vect) const {
  CheckInputDimension(x_vect);
  double x = x_vect[0];
  return Eigen::VectorXd::Constant(
      1, x <= margin_ ? std::numeric_limits<double>::infinity() : -log(x));
}

Eigen::MatrixXd LogBarrier::Jacobian(const Eigen::VectorXd& x_vect) const {
  CheckInputDimension(x_vect);
  double x = x_vect[0];
  return Eigen::MatrixXd::Constant(1, 1, x <= margin_ ? 0 : (-1. / x));
}

Eigen::MatrixXd LogBarrier::Hessian(const Eigen::VectorXd& x_vect) const {
  CheckSingleOutputDimension();
  CheckInputDimension(x_vect);
  double x = x_vect[0];
  return Eigen::MatrixXd::Constant(1, 1, x <= margin_ ? 0 : 1. / (x * x));
}

//------------------------------------------------------------------------------
// SoftNorm implementation
//------------------------------------------------------------------------------

Eigen::VectorXd SoftNorm::Forward(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  Eigen::VectorXd xd = x - x0_;
  double alpha_norm = sqrt(xd.transpose() * xd + alpha_sq_);
  return Eigen::VectorXd::Constant(1, alpha_norm - alpha_);
}

Eigen::MatrixXd SoftNorm::Jacobian(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  Eigen::VectorXd xd = x - x0_;
  double alpha_norm = sqrt(xd.transpose() * xd + alpha_sq_);
  return xd.transpose() / alpha_norm;
}

Eigen::MatrixXd SoftNorm::Hessian(const Eigen::VectorXd& x) const {
  CheckSingleOutputDimension();
  CheckInputDimension(x);
  Eigen::VectorXd xd = x - x0_;
  double alpha_norm = sqrt(xd.transpose() * xd + alpha_sq_);
  Eigen::VectorXd x_alpha_normalized = xd / alpha_norm;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_, n_);
  double gamma = 1. / alpha_norm;
  return gamma * (I - x_alpha_normalized * x_alpha_normalized.transpose());
}

//-----------------------------------------------------------------------------
// LogSumExp implementation.
//-----------------------------------------------------------------------------

LogSumExp::~LogSumExp() {}

Eigen::VectorXd LogSumExp::Forward(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  Eigen::VectorXd z = (alpha_ * x).array().exp();
  return Eigen::VectorXd::Constant(1, inv_alpha_ * std::log(z.sum()));
}

Eigen::MatrixXd LogSumExp::Jacobian(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  Eigen::VectorXd z = (alpha_ * x).array().exp();
  double z_sum = z.sum();
  return z.transpose() / z_sum;
}

Eigen::MatrixXd LogSumExp::Hessian(const Eigen::VectorXd& x) const {
  CheckSingleOutputDimension();
  CheckInputDimension(x);
  Eigen::MatrixXd H(Eigen::MatrixXd::Zero(n_, n_));
  Eigen::VectorXd z = (alpha_ * x).array().exp();
  double z_sum = z.sum();
  double p_inv = 1 / z_sum;
  Eigen::MatrixXd M = z.asDiagonal();
  H = p_inv * M - std::pow(p_inv, 2) * z * z.transpose();
  H *= alpha_;
  return H;
}

//-----------------------------------------------------------------------------
// NegLogSumExp implementation.
//-----------------------------------------------------------------------------

NegLogSumExp::~NegLogSumExp() {}

//-----------------------------------------------------------------------------
// Logistic implementation.
//-----------------------------------------------------------------------------

double Sigmoid(double x) {
  if (x > 0) {
    return 1. / (1. + exp(-x));
  } else {
    double expx = exp(x);
    return expx / (1. + expx);
  }
}

double LogisticF(double x, double k, double x0, double L) {
  return L * Sigmoid(k * (x - x0));
}

Logistic::~Logistic() {}

Eigen::VectorXd Logistic::Forward(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  y_[0] = LogisticF(x[0], k_, x0_, L_);
  return y_;
}

Eigen::MatrixXd Logistic::Jacobian(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  double p = Sigmoid(k_ * (x[0] - x0_));
  double v = L_ * p;
  J_(0, 0) = v * (1 - p) * k_;
  return J_;
}

Eigen::MatrixXd Logistic::Hessian(const Eigen::VectorXd& x) const {
  CheckSingleOutputDimension();
  CheckInputDimension(x);
  double p = Sigmoid(k_ * (x[0] - x0_));
  double v = L_ * p;
  J_(0, 0) = v * (1 - p) * k_;
  H_(0, 0) = J_(0, 0) * (1 - 2 * p) * k_;
  return H_;
}

//-----------------------------------------------------------------------------
// Arccos implementation.
//-----------------------------------------------------------------------------

Arccos::~Arccos() {}

// double a = 1. - x * x;
// *first_derivative = -1. / std::sqrt(a);
// *second_derivative = -x / std::pow(a, 1.5);

Eigen::VectorXd Arccos::Forward(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  y_[0] = std::acos(x[0]);
  return y_;
}

Eigen::MatrixXd Arccos::Jacobian(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  double a = 1. - x[0] * x[0];
  J_(0, 0) = -1. / std::sqrt(a);
  return J_;
}

Eigen::MatrixXd Arccos::Hessian(const Eigen::VectorXd& x) const {
  CheckSingleOutputDimension();
  CheckInputDimension(x);
  double a = 1. - x[0] * x[0];
  H_(0, 0) = -x[0] / std::pow(a, 1.5);
  return H_;
}

//-----------------------------------------------------------------------------
// Normalize implementation.
//-----------------------------------------------------------------------------

NormalizeMap::NormalizeMap(uint32_t n) : n_(n), min_norm_(1e-30) {
  type_ = "NormalizeMap";
}
NormalizeMap::~NormalizeMap() {}

Eigen::VectorXd NormalizeMap::Forward(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  return x / std::max(min_norm_, x.norm());
}

Eigen::MatrixXd NormalizeMap::Jacobian(const Eigen::VectorXd& x) const {
  CheckInputDimension(x);
  double x_norm = std::max(min_norm_, x.norm());
  double dinv = 1. / x_norm;
  Eigen::MatrixXd M = (dinv * Eigen::VectorXd::Ones(n_)).asDiagonal();
  return M - std::pow(dinv, 3) * x * x.transpose();
}
