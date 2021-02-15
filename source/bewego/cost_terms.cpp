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
#include <bewego/cost_terms.h>

using namespace bewego;

ObstaclePotential::ObstaclePotential(
    std::shared_ptr<const DifferentiableMap> sdf, double alpha,
    double rho_scaling) {
  signed_distance_field_ = sdf;
  ambient_space_dim_ = sdf->input_dimension();
  alpha_ = alpha;
  rho_scaling_ = rho_scaling;
}

Eigen::VectorXd ObstaclePotential::Forward(const Eigen::VectorXd& x) const {
  assert(x.size() == ambient_space_dim_);
  double sd = signed_distance_field_->Forward(x)[0];
  double v = rho_scaling_ * exp(-alpha_ * sd);
  return Eigen::VectorXd::Constant(1, v);
}

Eigen::MatrixXd ObstaclePotential::Jacobian(const Eigen::VectorXd& x) const {
  assert(x.size() == ambient_space_dim_);
  double rho = Forward(x)[0];
  return -alpha_ * rho * signed_distance_field_->Jacobian(x);
}

Eigen::MatrixXd ObstaclePotential::Hessian(const Eigen::VectorXd& x) const {
  assert(x.size() == ambient_space_dim_);
  // TODO...
  return Eigen::MatrixXd::Zero(ambient_space_dim_, ambient_space_dim_);
}
