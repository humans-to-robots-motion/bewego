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
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/util/misc.h>

#include <iostream>
using std::cout;
using std::endl;

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
  Eigen::MatrixXd J = Jacobian(x);
  Eigen::MatrixXd J_diff = FiniteDifferenceJacobian(*this, x);
  if (debug_) {
    cout << "J : " << endl << J << endl;
    cout << "J_diff : " << endl << J_diff << endl;
  }
  return J.isApprox(J_diff, precision);
}

/** check against finite differences */
bool DifferentiableMap::CheckHessian(double precision) const {
  Eigen::VectorXd x = Eigen::VectorXd::Random(input_dimension());
  Eigen::MatrixXd H = Hessian(x);
  Eigen::MatrixXd H_diff = FiniteDifferenceHessian(*this, x);
  if (debug_) {
    cout << "H : " << endl << H << endl;
    cout << "H_diff : " << endl << H_diff << endl;
  }
  return H.isApprox(H_diff, precision);
}

void DifferentialMapTest::FiniteDifferenceTest(
    std::shared_ptr<const DifferentiableMap> f,
    const Eigen::VectorXd& x) const {
  Eigen::MatrixXd J, J_diff;
  Eigen::MatrixXd H, H_diff;

  J = f->Jacobian(x);
  H = f->Hessian(x);

  J_diff = DifferentiableMap::FiniteDifferenceJacobian(*f, x);
  H_diff = DifferentiableMap::FiniteDifferenceHessian(*f, x);

  if (verbose_) {
    cout << "Test for  x : " << x.transpose() << endl;
    cout << "J : " << endl << J << endl;
    cout << "J_diff : " << endl << J_diff << endl;
    cout << "H : " << endl << H << endl;
    cout << "H_diff : " << endl << H_diff << endl;
  }

  if (use_relative_eq_) {
    EXPECT_TRUE(util::AlmostEqualRelative(J, J_diff, gradient_precision_));
    EXPECT_TRUE(util::AlmostEqualRelative(H, H_diff, hessian_precision_));
  } else {
    double max_J_delta = (J - J_diff).cwiseAbs().maxCoeff();
    double max_H_delta = (H - H_diff).cwiseAbs().maxCoeff();
    EXPECT_NEAR(max_J_delta, 0., gradient_precision_);
    EXPECT_NEAR(max_H_delta, 0., hessian_precision_);
  }
}

void DifferentialMapTest::AddRandomTests(
    std::shared_ptr<const DifferentiableMap> f, uint32_t n) {
  for (uint32_t i = 0; i < n; i++) {
    Eigen::VectorXd x = Eigen::VectorXd::Random(f->input_dimension());
    function_tests_.push_back(std::make_pair(f, x));
  }
}

void DifferentialMapTest::RunAllTests() const {
  for (uint32_t i = 0; i < function_tests_.size(); ++i) {
    auto test = function_tests_[i];
    FiniteDifferenceTest(test.first, test.second);
  }
}

}  // namespace bewego