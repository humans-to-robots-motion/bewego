// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/derivatives/combination_operators.h>
#include <bewego/motion/cost_terms.h>
#include <bewego/util/misc.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>

using namespace bewego;
using std::cout;
using std::endl;

std::shared_ptr<DifferentiableMap> f;
static const uint32_t NB_TESTS = 10;
static const unsigned int SEED = 0;

TEST_F(DifferentialMapTest, finite_differences_velocity) {
  std::srand(SEED);

  verbose_ = false;
  gradient_precision_ = 1e-6;
  hessian_precision_ = 1e-6;
  use_relative_eq_ = true;
  uint32_t n = 10;

  AddRandomTests(std::make_shared<FiniteDifferencesVelocity>(1, .01), n);
  AddRandomTests(std::make_shared<FiniteDifferencesVelocity>(2, .01), n);
  AddRandomTests(std::make_shared<FiniteDifferencesVelocity>(7, .1), n);
  RunAllTests();

  uint32_t N = 7;
  double dt = 0.1;
  FiniteDifferencesVelocity fd(N, dt);
  Eigen::VectorXd x_1 = Eigen::VectorXd::Random(N);
  Eigen::VectorXd x_2 = Eigen::VectorXd::Random(N);
  Eigen::VectorXd x_3(2 * N);
  x_3.head(N) = x_1;  // x_{t}
  x_3.tail(N) = x_2;  // x_{t+1}
  Eigen::VectorXd dx = fd(x_3);
  if (verbose_) {
    cout << "dx1 : " << dx.transpose() << endl;
    cout << "dx2 : " << ((x_2 - x_1) / dt).transpose() << endl;
  }
  ASSERT_TRUE(dx.isApprox((x_2 - x_1) / dt));

  f = std::make_shared<FiniteDifferencesVelocity>(7, .1);
  ASSERT_TRUE(f->type() == "FiniteDifferencesVelocity");
}

TEST_F(DifferentialMapTest, finite_differences_posvel) {
  std::srand(SEED);

  verbose_ = false;
  gradient_precision_ = 1e-6;
  hessian_precision_ = 1e-6;
  use_relative_eq_ = true;
  uint32_t n = 10;

  AddRandomTests(std::make_shared<FiniteDifferencesPosVel>(1, .01), n);
  AddRandomTests(std::make_shared<FiniteDifferencesPosVel>(2, .01), n);
  AddRandomTests(std::make_shared<FiniteDifferencesPosVel>(7, .1), n);
  RunAllTests();

  uint32_t N = 7;
  double dt = 0.1;
  FiniteDifferencesPosVel fd(N, dt);
  Eigen::VectorXd x_1 = Eigen::VectorXd::Random(N);
  Eigen::VectorXd x_2 = Eigen::VectorXd::Random(N);
  Eigen::VectorXd x_3(2 * N);
  x_3.head(N) = x_1;  // x_{t}
  x_3.tail(N) = x_2;  // x_{t+1}
  Eigen::VectorXd dx = fd(x_3);
  if (verbose_) {
    cout << "dx1 : " << dx.transpose() << endl;
    cout << "dx2 : " << ((x_2 - x_1) / dt).transpose() << endl;
  }
  ASSERT_TRUE(dx.head(N).isApprox(x_1));
  ASSERT_TRUE(dx.tail(N).isApprox((x_2 - x_1) / dt));

  f = std::make_shared<FiniteDifferencesPosVel>(7, .1);
  ASSERT_TRUE(f->type() == "FiniteDifferencesPosVel");
}

TEST_F(DifferentialMapTest, finite_differences_acceleration) {
  std::srand(SEED);

  verbose_ = false;
  gradient_precision_ = 1e-6;
  hessian_precision_ = 1e-6;
  use_relative_eq_ = true;
  uint32_t n = 10;

  AddRandomTests(std::make_shared<FiniteDifferencesAcceleration>(1, .01), n);
  AddRandomTests(std::make_shared<FiniteDifferencesAcceleration>(2, .01), n);
  AddRandomTests(std::make_shared<FiniteDifferencesAcceleration>(7, .1), n);
  RunAllTests();

  uint32_t N = 7;
  double dt = 0.1;
  FiniteDifferencesAcceleration fd(N, dt);
  Eigen::VectorXd x_1 = Eigen::VectorXd::Random(N);
  Eigen::VectorXd x_2 = Eigen::VectorXd::Random(N);
  Eigen::VectorXd x_3 = Eigen::VectorXd::Random(N);
  Eigen::VectorXd x_4 = Eigen::VectorXd::Random(3 * N);
  x_4.head(N) = x_1;        // x_{t-1}
  x_4.segment(N, N) = x_2;  // x_{t}
  x_4.tail(N) = x_3;        // x_{t+1}
  Eigen::VectorXd dx = fd(x_4);
  ASSERT_TRUE(dx.isApprox((x_1 + x_3 - 2 * x_2) / (.1 * .1)));

  f = std::make_shared<FiniteDifferencesAcceleration>(7, .1);
  ASSERT_TRUE(f->type() == "FiniteDifferencesAcceleration");
}

TEST(cost_terms, squared_norm_velocity) {
  std::srand(SEED);

  f = std::make_shared<SquaredNormVelocity>(1, .01);
  // f->set_debug(true);
  ASSERT_TRUE(f->CheckJacobian());
  ASSERT_TRUE(f->CheckHessian());

  f = std::make_shared<SquaredNormVelocity>(7, .1);
  // f->set_debug(true);
  ASSERT_TRUE(f->CheckJacobian(1e-6));
  ASSERT_TRUE(f->CheckHessian(1e-6));

  Eigen::VectorXd x_1 = Eigen::VectorXd::Random(7);
  Eigen::VectorXd x_2 = Eigen::VectorXd::Random(7);
  Eigen::VectorXd x_3 = Eigen::VectorXd::Random(2 * 7);
  x_3.head(7) = x_1;  // x_{t}
  x_3.tail(7) = x_2;  // x_{t+1}
  double sq_norm_1 = (*f)(x_3)[0] * 2;
  double sq_norm_2 = ((x_2 - x_1) / .1).squaredNorm();
  EXPECT_NEAR(sq_norm_1, sq_norm_2, 1e-3);

  ASSERT_TRUE(f->type() == "SquaredNormVelocity");
}

TEST(cost_terms, compose) {
  std::srand(SEED);

  uint32_t dim = 1;
  double dt = .01;

  DifferentiableMapPtr f1 = std::make_shared<SquaredNormVelocity>(dim, dt);
  ASSERT_TRUE(f1->CheckJacobian());
  ASSERT_TRUE(f1->CheckHessian());

  DifferentiableMapPtr f2 = std::make_shared<Compose>(
      std::make_shared<SquaredNorm>(dim),
      std::make_shared<FiniteDifferencesVelocity>(dim, dt));

  ASSERT_TRUE(f2->CheckJacobian(1e-6));
  ASSERT_TRUE(f2->CheckHessian());

  Eigen::VectorXd x_1 = Eigen::VectorXd::Random(dim);
  Eigen::VectorXd x_2 = Eigen::VectorXd::Random(dim);
  Eigen::VectorXd x_3 = Eigen::VectorXd::Random(2 * dim);
  x_3.head(dim) = x_1;  // x_{t}
  x_3.tail(dim) = x_2;  // x_{t+1}

  double sq_norm_1 = (*f1)(x_3)[0];
  double sq_norm_2 = (*f2)(x_3)[0];
  EXPECT_NEAR(sq_norm_1, sq_norm_2, 1e-12);

  ASSERT_TRUE(f1->type() == "SquaredNormVelocity");
  ASSERT_TRUE(f2->type() == "Compose");
}

TEST_F(DifferentialMapTest, bound_barrier) {
  std::srand(SEED);
  verbose_ = false;
  gradient_precision_ = 1e-3;
  hessian_precision_ = 1e-2;
  use_relative_eq_ = true;
  Eigen::Vector3d v_lower(-1, -1, -1);
  Eigen::Vector3d v_upper(1, 1, 1);
  auto phi = std::make_shared<BoundBarrier>(v_lower, v_upper);
  AddRandomTests(phi, NB_TESTS);
  RunAllTests();

  ASSERT_TRUE(phi->type() == "BoundBarrier");
}

TEST_F(DifferentialMapTest, pos_vel_differentiable_map) {
  std::srand(SEED);
  verbose_ = false;
  gradient_precision_ = 1e-6;
  hessian_precision_ = -1;
  auto f = std::make_shared<TrigoTestMap>();
  auto phi = std::make_shared<PosVelDifferentiableMap>(f);
  AddRandomTests(phi, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi->type() == "PosVelDifferentiableMap");
}
