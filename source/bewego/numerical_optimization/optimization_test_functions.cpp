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
 *                                              Jim Mainprice We 18 Dec 2019
 */

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/numerical_optimization/optimization_test_functions.h>
#include <bewego/util/misc.h>
#include <filesystem>
namespace fs = std::__fs::filesystem;

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace bewego::util;
using namespace std;
using std::cout;
using std::endl;

void QuadricProgramOptimizationTest::SetUp() {
  srand(101);
  uint32_t n = 10;            // Dimension of the domain.
  uint32_t k_inequality = 5;  // Number of inequality constraints.
  uint32_t k_equality = 2;    // Number of equality constraints.

  Eigen::MatrixXd H = Eigen::MatrixXd::Random(n, n);
  H = H * H.transpose();  // Make sure H is symetric (and positive definite)
  Eigen::VectorXd d = Eigen::VectorXd::Random(n);
  double c = 2.;

  A_ = Eigen::MatrixXd::Random(k_inequality, n);
  a_ = Eigen::VectorXd::Random(k_inequality);
  B_ = Eigen::MatrixXd::Random(k_equality, n);
  b_ = Eigen::VectorXd::Random(k_equality);

  cout << "CURRENT PATH : " <<  fs::current_path() << endl;

  // util::ReadMatrixFromCsvFile

  if (verbose_) {
    cout << "H  : " << endl << H << endl;
    cout << "d  : " << endl << d << endl;
    cout << "c  : " << endl << c << endl;
    cout << "A_ : " << endl << A_ << endl;
    cout << "a_ : " << endl << a_ << endl;
    cout << "B_ : " << endl << B_ << endl;
    cout << "b_ : " << endl << b_ << endl;
  }

  // Set objective
  f_ = std::make_shared<QuadricMap>(H, d, c);
  qp_ = std::make_shared<QuadraticProgram>(*f_, A_, a_, B_, b_, false);
  qp_sel_ = std::make_shared<QuadraticProgram>(*f_, A_, a_, B_, b_, true);
  x0_ = Eigen::VectorXd::Zero(n);
}

void bewego::numerical_optimization::ExpectNear(
    const Eigen::VectorXd& x, const Eigen::VectorXd& x_expected,
    double precision, bool verbose) {
  if (verbose) {
    cout << " --- vector near (" << precision << ") :" << endl;
    cout << x.transpose() << endl;
    cout << x_expected.transpose() << endl;
    cout << " dist : " << (x - x_expected).norm() << endl;
  }

  EXPECT_EQ(x.size(), x_expected.size());
  EXPECT_NEAR((x - x_expected).norm(), 0., precision);
}

// Use this solution validation for the linear one-sided quadratic inequality
// penalties. This solution isn't actually as good as the one found by the
// selective penalty term. (See below.)
void QuadricProgramOptimizationTest::ValidateSolution(
    const ConstrainedSolution& solution) const {
  // TODO
}

void GenericQuadricProgramTest::SetUp() {
  QuadricProgramOptimizationTest::SetUp();

  // Check that sizes don't not overflow
  n_g_ = size_t_to_uint(a_.size());
  n_h_ = size_t_to_uint(b_.size());

  // Set inequality constraints
  g_constraints_.resize(n_g_);
  for (uint32_t i = 0; i < n_g_; i++) {
    g_constraints_[i] = std::make_shared<AffineMap>(-A_.row(i), a_[i]);
  }

  // Set equality constraints
  h_constraints_.resize(n_h_);
  for (uint32_t i = 0; i < n_h_; i++) {
    h_constraints_[i] = std::make_shared<AffineMap>(-B_.row(i), b_[i]);
  }

  nonlinear_problem_ = std::make_shared<OptimizationProblemWithConstraints>(
      f_, g_constraints_, h_constraints_);
}

// Use this solution validation for the linear one-sided quadratic inequality
// penalties. This solution isn't actually as good as the one found by the
// selective penalty term. (See below.)
void GenericQuadricProgramTest::ValidateSolution(
    const ConstrainedSolution& solution) const {
  // TODO
}

void GenericQuadricalyConstrainedQuadricProgramTest::SetUp() {
  srand(101);
  uint32_t n = 10;            // Dimension of the domain.
  uint32_t k_inequality = 1;  // Number of inequality constraints.
  uint32_t k_equality = 2;    // Number of equality constraints.

  Eigen::MatrixXd H = Eigen::MatrixXd::Random(n, n);
  H = H * H.transpose();  // Make sure H is symetric (and positive definite)
  Eigen::VectorXd d = Eigen::VectorXd::Random(n);
  double c = 2.;

  A_ = Eigen::MatrixXd::Random(n, n);
  A_ = A_ * A_.transpose();  // Make sure H is symetric (and positive definite)
  a_ = Eigen::VectorXd::Random(n);
  double a = 2.;

  B_ = Eigen::MatrixXd::Random(k_equality, n);
  b_ = Eigen::VectorXd::Random(k_equality);

  if (verbose_) {
    cout << "H : " << endl << H << endl;
    cout << "d : " << endl << d << endl;
    cout << "c : " << endl << c << endl;
    cout << "A_ : " << endl << A_ << endl;
    cout << "a_ : " << endl << a_ << endl;
    cout << "a : " << endl << a << endl;
    cout << "B_ : " << endl << B_ << endl;
    cout << "b_ : " << endl << b_ << endl;
  }

  // Set objective
  f_ = std::make_shared<QuadricMap>(H, d, c);
  g_ = std::make_shared<QuadricMap>(A_, a_, a);
  x0_ = Eigen::VectorXd::Zero(n);

  // Check that sizes don't not overflow
  n_g_ = k_inequality;
  n_h_ = size_t_to_uint(b_.size());

  // Set inequality constraints
  g_constraints_.resize(n_g_);
  g_constraints_[0] = g_;

  // Set equality constraints
  h_constraints_.resize(n_h_);
  for (uint32_t i = 0; i < n_h_; i++) {
    h_constraints_[i] = std::make_shared<AffineMap>(-B_.row(i), b_[i]);
  }

  nonlinear_problem_ = std::make_shared<OptimizationProblemWithConstraints>(
      f_, g_constraints_, h_constraints_);
}

// Use this solution validation for the linear one-sided quadratic inequality
// penalties. This solution isn't actually as good as the one found by the
// selective penalty term. (See below.)
void GenericQuadricalyConstrainedQuadricProgramTest::ValidateSolution(
    const ConstrainedSolution& solution) const {}
