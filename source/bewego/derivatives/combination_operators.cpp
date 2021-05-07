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
 *                                                             Thu 1 Apr 2021
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/derivatives/combination_operators.h>

using namespace bewego;
using std::cout;
using std::endl;

//------------------------------------------------------------------------------
// SoftDist implementation
//------------------------------------------------------------------------------

SoftDist::SoftDist(DifferentiableMapPtr sq_dist, double alpha)
    : sq_dist_(sq_dist), alpha_(alpha), alpha_sq_(alpha * alpha) {
  assert(sq_dist_.get() != nullptr);
  assert(sq_dist_->output_dimension() == 1);
  type_ = "SoftDist";
}

Eigen::VectorXd SoftDist::Forward(const Eigen::VectorXd& x) const {
  double alpha_norm = sqrt(sq_dist_->ForwardFunc(x) + alpha_sq_);
  return Eigen::VectorXd::Constant(1, alpha_norm - alpha_);
}

Eigen::MatrixXd SoftDist::Jacobian(const Eigen::VectorXd& x) const {
  assert(x.size() == n_);
  double alpha_norm = sqrt(sq_dist_->ForwardFunc(x) + alpha_sq_);
  return 0.5 * sq_dist_->Jacobian(x) / alpha_norm;
}

Eigen::MatrixXd SoftDist::Hessian(const Eigen::VectorXd& x) const {
  uint32_t n = input_dimension();
  double alpha_norm = sqrt(sq_dist_->ForwardFunc(x) + alpha_sq_);
  Eigen::VectorXd x_alpha_normalized = 0.5 * sq_dist_->Gradient(x) / alpha_norm;
  Eigen::MatrixXd H_d = .5 * sq_dist_->Hessian(x);
  double gamma = 1. / alpha_norm;
  return gamma * (H_d - x_alpha_normalized * x_alpha_normalized.transpose());
}

//------------------------------------------------------------------------------
// LogBarrierWithApprox implementation
//------------------------------------------------------------------------------

Eigen::VectorXd LogBarrierWithApprox::Forward(
    const Eigen::VectorXd& x_vect) const {
  assert(x_vect.size() == 1);
  double x = x_vect[0];
  double v = 0;
  if (x <= 0) {
    v = std::numeric_limits<double>::infinity();
  } else {
    if (x <= x_splice_) {
      return approximation_->Forward(x_vect);
    } else {
      v = -scalar_ * log(x);
    }
  }
  return Eigen::VectorXd::Constant(1, v);
}

Eigen::MatrixXd LogBarrierWithApprox::Jacobian(
    const Eigen::VectorXd& x_vect) const {
  assert(x_vect.size() == 1);
  double x = x_vect[0];
  double g = 0;
  if (x > 0) {
    g = x <= x_splice_ ? approximation_->Jacobian(x_vect)(0, 0) : -scalar_ / x;
  }
  return Eigen::MatrixXd::Constant(1, 1, g);
}

Eigen::MatrixXd LogBarrierWithApprox::Hessian(
    const Eigen::VectorXd& x_vect) const {
  assert(x_vect.size() == 1);
  double x = x_vect[0];
  double h = 0;
  if (x > 0) {
    h = x <= x_splice_ ? approximation_->Hessian(x_vect)(0, 0)
                       : scalar_ / (x * x);
  }
  return Eigen::MatrixXd::Constant(1, 1, h);
}

std::shared_ptr<DifferentiableMap> LogBarrierWithApprox::MakeTaylorLogBarrier()
    const {
  auto log_barrier = std::make_shared<LogBarrier>();
  auto scaled_log_barrier = std::make_shared<Scale>(log_barrier, scalar_);
  return std::make_shared<SecondOrderTaylorApproximation>(
      *scaled_log_barrier, Eigen::VectorXd::Constant(1, x_splice_));
}

//-----------------------------------------------------------------------------
// DotProduct implementation.
//-----------------------------------------------------------------------------

DotProduct::DotProduct(DifferentiableMapPtr map1, DifferentiableMapPtr map2)
    : map1_(map1), map2_(map2), n_(map1->input_dimension()) {
  assert(map1->input_dimension() == map2->input_dimension());
  assert(map1->output_dimension() == map2->output_dimension());
}

Eigen::VectorXd DotProduct::Forward(const Eigen::VectorXd& x) const {
  assert(x.size() == input_dimension());
  Eigen::VectorXd x1 = (*map1_)(x);
  Eigen::VectorXd x2 = (*map2_)(x);
  return Eigen::VectorXd::Constant(1, x1.transpose() * x2);
}

Eigen::MatrixXd DotProduct::Jacobian(const Eigen::VectorXd& x) const {
  assert(x.size() == input_dimension());
  Eigen::VectorXd x1 = (*map1_)(x);
  Eigen::VectorXd x2 = (*map2_)(x);
  Eigen::MatrixXd J1 = map1_->Jacobian(x);
  Eigen::MatrixXd J2 = map2_->Jacobian(x);
  return J1 * x2 + J2 * x1;
}

Eigen::MatrixXd DotProduct::Hessian(const Eigen::VectorXd& x) const {
  assert(x.size() == input_dimension());
  Eigen::MatrixXd J1 = map1_->Jacobian(x);
  Eigen::MatrixXd J2 = map2_->Jacobian(x);
  return J1 * J2.transpose() + J2 * J1.transpose();
}