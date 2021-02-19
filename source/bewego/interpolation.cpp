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
#include <bewego/interpolation.h>

#include <Eigen/Core>
#include <Eigen/LU>  // for matrix inverse
#include <iostream>

using std::cout;
using std::endl;

namespace bewego {
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

  // Compute weighted points: WX, where W is the diagonal matrix of weights.
  Eigen::MatrixXd WX(Xaug.rows(), Xaug.cols());
  Eigen::VectorXd diff(x_query.size());
  Eigen::MatrixXd D_scale = -.5 * D;
  for (int i = 0; i < X.rows(); i++) {
    diff = X.row(i).transpose() - x_query;
    WX.row(i) = exp( diff.transpose() * D_scale * diff) * Xaug.row(i);
  }

  // Fit plane to the weighted data
  // Calculate Pinv = X'WX + lambda I. P = inv(Pinv) is then
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