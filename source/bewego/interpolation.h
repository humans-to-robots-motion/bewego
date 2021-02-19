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
#pragma once

#include <bewego/differentiable_map.h>

namespace bewego {

/**
 Calculates the linear regression given a dataset

 Parameters:
   X's rows contain data points.
   Y's entries contain corresponding targets.

 Calculates:
 w^* = \argmin 1/2 |Y - w^T X|^2 + lambda_1/2 |w|^2 + lambda_2/2 |w - w_t|^2

 Solution:
 w^* = (X'X + (lambda_1+lambda_2) I)^{-1} (X'Y + lambda_2 w_t)
*/
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

class LinearRegressor {
 public:
  LinearRegressor() : lambda_1_(0.), lambda_2_(0.) {}
  LinearRegressor(double l1, double l2) : lambda_1_(l1), lambda_2_(l2) {}
  LinearRegressor(double l1) : lambda_1_(l1), lambda_2_(0.) {}

  // Compute the hyperplane
  Eigen::VectorXd ComputeParameters(const Eigen::MatrixXd& X,
                                    const Eigen::VectorXd& Y,
                                    const Eigen::VectorXd& w_t) const {
    return CalculateLinearRegression(X, Y, w_t, lambda_1_, lambda_2_);
  }

  // Compute the hyperplane
  // in this case only the L2 norm is considered unless set to 0.
  Eigen::VectorXd ComputeParameters(const Eigen::MatrixXd& X,
                                    const Eigen::VectorXd& Y) const {
    Eigen::VectorXd w_t(Eigen::VectorXd::Zero(X.cols()));
    return CalculateLinearRegression(X, Y, w_t, lambda_1_, 0.);
  }

  void set_lambda_1(double l1) { lambda_1_ = l1; }
  void set_lambda_2(double l2) { lambda_2_ = l2; }

  double lambda_1() const { return lambda_1_; }
  double lambda_2() const { return lambda_2_; }

 private:
  double lambda_1_;  // L2 norm scalar
  double lambda_2_;  // Proximal scalar
};

/**
VectorXd x_query
MatrixXd X
VectorXd Y
MatrixXd D
double ridge_lambda

Calculates the locally weighted regression at the query point.

 Parameters:
  x_query is a column vector with the query point.
  X's rows contain domain points.
  Y's entries contain corresponding targets.
  D gives the Mahalanobis metric as:
  dist(x_query, x) = sqrt( (x_query - x)'D(x_query - x) )
  ridge_lambda is the regression regularizer, denoted lambda in the
  calculation below

  Calculates:
      beta^* = argmin 1/2 |Y - X beta|_W^2 + lambda/2 |w|^2

      with W a diagonal matrix with elements
        w_i = \exp{ -1/2 |x_query - x_i|_D^2

  Solution: beta^* = inv(X'WX + lambda I)X'WY.
  Final returned value: beta^*'x_query.

  Note that all points are augmented with an extra
*/
double CalculateLocallyWeightedRegression(const Eigen::VectorXd& x_query,
                                          const Eigen::MatrixXd& X,
                                          const Eigen::VectorXd& Y,
                                          const Eigen::MatrixXd& D,
                                          double ridge_lambda);

class LWR : public DifferentiableMap {
 public:
  LWR(uint32_t m, uint32_t n) : m_(m), n_(n) {}
  uint32_t input_dimension() const { return n_; }
  uint32_t output_dimension() const { return m_; }
  
  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    assert(m_ == X_.size());
    assert(m_ == Y_.size());
    assert(m_ == D_.size());
    assert(m_ == ridge_lambda_.size());
    Eigen::VectorXd y(m_);
    for (uint32_t i = 0; i < m_; i++) {
      y[i] = CalculateLocallyWeightedRegression(x, X_[i], Y_[i], D_[i],
                                                ridge_lambda_[i]);
    }
    return y;
  }

  std::vector<Eigen::VectorXd> ForwardMultiQuerry(
      const std::vector<Eigen::VectorXd>& xs) const;

  std::vector<Eigen::MatrixXd> JacobianMultiQuerry(
      const std::vector<Eigen::VectorXd>& xs) const;

  std::vector<Eigen::MatrixXd> X_;
  std::vector<Eigen::VectorXd> Y_;
  std::vector<Eigen::MatrixXd> D_;
  std::vector<double> ridge_lambda_;

 protected:
  uint32_t m_;
  uint32_t n_;
};

}  // namespace bewego