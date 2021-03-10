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
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/numerical_optimization/trajectory_optimization.h>
#include <bewego/util/chrono.h>
#include <bewego/util/misc.h>
#include <gtest/gtest.h>

#include <Eigen/Sparse>

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace std;
using std::cout;
using std::endl;

typedef Eigen::SparseMatrix<double>
    SpMat;  // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

TEST(SparseMatrix, solve) {
  // void buildProblem(std::vector<T> & coefficients, Eigen::VectorXd & b, int
  // n);

  ChronoOn();

  int n = 300;    // size of the image
  int m = n * n;  // number of unknows (=number of pixels)

  // Assembly:
  std::vector<T> coefficients;  // list of non-zeros coefficients
  Eigen::VectorXd b(m);
  // the right hand side-vector resulting from the constraints
  // buildProblem(coefficients, b, n);

  SpMat A(m, m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(
      A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(
      b);  // use the factorization to solve for the given right hand side

  ChronoPrint("Solving sparse matrix");
  ChronoOff();
}

TEST_F(TrajectoryQCQPTest, hessian) {
  verbose_ = false;

  // Simple trajectory optimization problem defined
  q_init_ = Eigen::Vector2d(0, 0);
  q_goal_ = Eigen::Vector2d(1, 1);
  T_ = 10;
  double dt = 0.01;
  uint32_t n = util::size_t_to_uint(q_init_.size());
  double scalar_acc = 1e-5;
  double scalar_goal = 1;
  auto problem = std::make_shared<TrajectoryObjectiveTest>(
      n, dt, T_, scalar_acc, scalar_goal, q_init_, q_goal_);

  // Zero motion trajectory
  initial_solution_ = Trajectory(n, T_);
  for (uint32_t t = 0; t <= T_ + 1; ++t) {
    initial_solution_.Configuration(t) = q_init_;
  }

  auto objective = problem->ObjectiveDiffFunction();
  auto x_rand1 = util::Random(objective->input_dimension());
  auto x_rand2 = util::Random(objective->input_dimension());

  Eigen::MatrixXd H =
      objective->Hessian(x_rand1) +
      1e-6 * Eigen::MatrixXd::Identity(objective->input_dimension(),
                                       objective->input_dimension());

  if (verbose_) {
    cout << "H : " << endl << H << endl;
    // cout << "H : " << endl << H.inverse() << endl;
  }
  ChronoOn();

  Eigen::VectorXd dx1, dx2;

  for (uint32_t i = 0; i < 1000; i++) {
    dx1 = H.ldlt().solve(x_rand2);
  }

  ChronoPrint("Hessian 1");

  for (uint32_t i = 0; i < 1000; i++) {
    dx2 = H.inverse() * x_rand2;
  }

  ChronoPrint("Hessian 2");

  ChronoOff();

  if (verbose_) {
    cout << "dx1 : " << dx1.transpose() << endl;
    cout << "dx2 : " << dx2.transpose() << endl;
  }

  EXPECT_TRUE(util::AlmostEqualRelative(dx1, dx2));
}

TEST_F(TrajectoryQCQPTest, motion_sparsity_patern) {
  verbose_ = false;

  // Simple trajectory optimization problem defined
  uint32_t n = 7;
  q_init_ = Eigen::VectorXd::Zero(n);
  q_goal_ = Eigen::VectorXd::Ones(n);
  T_ = 30;
  double dt = 0.01;
  double scalar_acc = 1e-5;
  double scalar_goal = 1;
  auto problem = std::make_shared<TrajectoryObjectiveTest>(
      n, dt, T_, scalar_acc, scalar_goal, q_init_, q_goal_);

  auto objective = problem->ObjectiveDiffFunction();
  uint32_t dim = objective->input_dimension();

  // Patern
  auto H_patern1 = objective->HessianSparcityPatern().Matrix(dim, dim);
  if (verbose_) {
    cout << "H_patern1 : " << endl << H_patern1 << endl;
  }

  // Ground truth
  Eigen::MatrixXi H_patern2 = Eigen::MatrixXi::Zero(dim, dim);
  for (int32_t i = 0; i < n * 3; i++) {
    int size = H_patern2.diagonal(i).size();
    auto ones = Eigen::VectorXi::Ones(size);
    H_patern2.diagonal(i) = ones;
    H_patern2.diagonal(-i) = ones;
  }
  if (verbose_) {
    cout << "H_patern2 : " << endl << H_patern2 << endl;
  }

  EXPECT_LT((H_patern1 - H_patern2).norm(), 1e-6);
}
