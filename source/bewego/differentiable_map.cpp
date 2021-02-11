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
 *                                                              Wed 4 Feb 2019
 */
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/differentiable_map.h>

namespace bewego {

Eigen::MatrixXd DifferentiableMap::FiniteDifferenceJacobian(
    const DifferentiableMap& f, const Eigen::VectorXd& q) {
  assert(q.size() == f.input_dimension());
  double dt = 1e-4;
  double dt_half = dt / 2.;
  Eigen::MatrixXd J =
      Eigen::MatrixXd::Zero(f.output_dimension(), f.input_dimension());

  for (uint32_t j = 0; j < q.size(); j++) {
    Eigen::VectorXd q_up = q;
    q_up[j] += dt_half;
    Eigen::VectorXd x_up = f(q_up);

    Eigen::VectorXd q_down = q;
    q_down[j] -= dt_half;
    Eigen::VectorXd x_down = f(q_down);

    J.col(j) = (x_up - x_down) / dt;
  }
  return J;
}

/**
    Takes an object f that has a forward method returning
    a numpy array when querried.
    */
Eigen::MatrixXd DifferentiableMap::FiniteDifferenceHessian(
    const DifferentiableMap& f, const Eigen::VectorXd& q) {
  assert(q.size() == f.input_dimension());
  assert(f.output_dimension() == 1);
  double dt = 1e-4;
  double dt_half = dt / 2.;
  Eigen::MatrixXd H =
      Eigen::MatrixXd::Zero(f.input_dimension(), f.input_dimension());

  for (uint32_t j = 0; j < q.size(); j++) {
    Eigen::VectorXd q_up = q;
    q_up[j] += dt_half;
    Eigen::VectorXd g_up = f.Gradient(q_up);

    Eigen::VectorXd q_down = q;
    q_down[j] -= dt_half;
    Eigen::VectorXd g_down = f.Gradient(q_down);

    H.col(j) = (g_up - g_down) / dt;
  }
  return H;
}

/** check against finite differences */
bool DifferentiableMap::CheckJacobian(double precision) const {
  Eigen::VectorXd x = Eigen::VectorXd::Random(input_dimension());
  return Jacobian(x).isApprox(FiniteDifferenceJacobian(*this, x), precision);
}

/** check against finite differences */
bool DifferentiableMap::CheckHessian(double precision) const {
  Eigen::VectorXd x = Eigen::VectorXd::Random(input_dimension());
  return Hessian(x).isApprox(FiniteDifferenceHessian(*this, x), precision);
}

}  // namespace bewego