/*
 * Copyright (c) 2019
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
#include <bewego/util/interpolation.h>

#include <Eigen/Core>
#include <Eigen/LU>  // for matrix inverse
#include <iostream>

using std::cout;
using std::endl;

namespace bewego {

Eigen::VectorXd CalculateLinearRegression(const Eigen::MatrixXd& X,
                                          const Eigen::VectorXd& Y,
                                          const Eigen::VectorXd& w_t,
                                          double lambda_1, double lambda_2) {
  assert(w_t.size() == X.cols());
  assert(X.rows() == Y.size());

  double lambda_ridge = lambda_1 + lambda_2;
  Eigen::MatrixXd diag =
      (lambda_ridge * Eigen::VectorXd::Ones(X.cols())).asDiagonal();
  Eigen::VectorXd beta = (X.transpose() * Y + lambda_2 * w_t);
  return (X.transpose() * X + diag).inverse() * beta;
}

double CalculateLocallyWeightedRegression(const Eigen::VectorXd& x_query,
                                          const Eigen::VectorXd& x_query_aug,
                                          const Eigen::MatrixXd& X,
                                          const Eigen::MatrixXd& Xaug,
                                          const Eigen::VectorXd& Y,
                                          const Eigen::MatrixXd& D_scale,
                                          double ridge_lambda) {
  Eigen::MatrixXd WX(Xaug.rows(), Xaug.cols());
  Eigen::VectorXd diff(x_query.size());

  // Compute weighted points: WX, where W is the diagonal matrix of weights.
  for (int i = 0; i < X.rows(); i++) {
    diff = X.row(i).transpose() - x_query;
    WX.row(i) = exp(diff.transpose() * D_scale * diff) * Xaug.row(i);
  }

  // Fit plane to the weighted data
  // Calculate Pinv = X'WX + lambda I. P = inv(Pinav) is then
  // P = inv(X'WX + lambda I).
  Eigen::MatrixXd Pinv = WX.transpose() * Xaug;
  if (ridge_lambda > 0) {
    for (int i = 0; i < Xaug.cols(); i++) {
      Pinv(i, i) += ridge_lambda;
    }
  }
  // beta = inv(X'WX + lambda I)WX'Y
  // Return inner product between plane and querrie point
  return (Pinv.inverse() * WX.transpose() * Y).transpose() * x_query_aug;
}

double CalculateLocallyWeightedRegression(const Eigen::VectorXd& x_query,
                                          const Eigen::MatrixXd& X,
                                          const Eigen::VectorXd& Y,
                                          const Eigen::MatrixXd& D,
                                          double ridge_lambda) {
  // Default value is 0. The calculation uses ridge regression with a finite
  // regularizer, so the values should diminish smoothly to 0 away from the
  // data set anyway.
  if (Y.size() == 0) return 0.;

  // The "augmented" version of X has an extra constant feature
  // to represent the bias.
  Eigen::MatrixXd Xaug(X.rows(), X.cols() + 1);
  Xaug << X, Eigen::VectorXd::Ones(Xaug.rows());

  Eigen::VectorXd x_query_aug(x_query.size() + 1);
  x_query_aug << x_query, 1;

  // Precompute the scaled metric
  Eigen::MatrixXd D_scale = -.5 * D;

  return CalculateLocallyWeightedRegression(x_query, x_query_aug, X, Xaug, Y,
                                            D_scale, ridge_lambda);
}

void LWR::Initialize(const std::vector<Eigen::MatrixXd>& X,
                     const std::vector<Eigen::VectorXd>& Y,
                     const std::vector<Eigen::MatrixXd>& D,
                     const std::vector<double> ridge_lambda) {
  assert(!X.empty());
  assert(!Y.empty());
  assert(!D.empty());

  X_ = X;  // Input data
  Y_ = Y;  // Targets
  D_ = D;  // Metrics

  Xaug_.resize(X.size());
  D_scale_.resize(D.size());

  // The "augmented" version of X has an extra constant feature
  // to represent the bias.
  for (uint32_t i = 0; i < Xaug_.size(); i++) {
    Xaug_[i] = Eigen::MatrixXd(X[i].rows(), X[i].cols() + 1);
    Xaug_[i] << X[i], Eigen::VectorXd::Ones(Xaug_[i].rows());
  }

  // Precompute the scaled metrics
  for (uint32_t i = 0; i < D_scale_.size(); i++) {
    D_scale_[i] = -.5 * D[i];
  }

  initialized_ = true;
}

Eigen::VectorXd LWR::Forward(const Eigen::VectorXd& x) const {
  assert(input_dimension() == x.size());
  assert(m_ == X_.size());
  assert(m_ == Y_.size());
  assert(m_ == D_.size());
  assert(m_ == ridge_lambda_.size());
  Eigen::VectorXd y(m_);
  if (!initialized_) {
    for (uint32_t i = 0; i < m_; i++) {
      y[i] = CalculateLocallyWeightedRegression(x, X_[i], Y_[i], D_[i],
                                                ridge_lambda_[i]);
    }
  } else {
    Eigen::VectorXd x_query_aug(x.size() + 1);
    x_query_aug << x, 1;
    for (uint32_t i = 0; i < m_; i++) {
      y[i] = CalculateLocallyWeightedRegression(x, x_query_aug, X_[i], Xaug_[i],
                                                Y_[i], D_scale_[i],
                                                ridge_lambda_[i]);
    }
  }
  return y;
}

std::vector<Eigen::VectorXd> LWR::ForwardMultiQuerry(
    const std::vector<Eigen::VectorXd>& xs) const {
  std::vector<Eigen::VectorXd> ys(xs.size());
  for (uint32_t i = 0; i < ys.size(); i++) {
    ys[i] = Forward(xs[i]);
  }
  return ys;
}

std::vector<Eigen::MatrixXd> LWR::JacobianMultiQuerry(
    const std::vector<Eigen::VectorXd>& xs) const {
  std::vector<Eigen::MatrixXd> Js(xs.size());
  for (uint32_t i = 0; i < Js.size(); i++) {
    Js[i] = Jacobian(xs[i]);
  }
  return Js;
}

}  // namespace bewego