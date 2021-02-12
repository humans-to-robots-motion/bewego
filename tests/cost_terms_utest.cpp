// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/cost_terms.h>
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

TEST(cost_terms, finite_differences_velocity) {
  std::srand(SEED);

  f = std::make_shared<FiniteDifferencesVelocity>(1, .01);
  ASSERT_TRUE(f->CheckJacobian());
  ASSERT_TRUE(f->CheckHessian());

  f = std::make_shared<FiniteDifferencesVelocity>(2, .01);
  ASSERT_TRUE(f->CheckJacobian());

  f = std::make_shared<FiniteDifferencesVelocity>(7, .1);
  ASSERT_TRUE(f->CheckJacobian());

  Eigen::VectorXd x_1 = Eigen::VectorXd::Random(7);
  Eigen::VectorXd x_2 = Eigen::VectorXd::Random(7);
  Eigen::VectorXd x_3 = Eigen::VectorXd::Random(2 * 7);
  x_3.head(7) = x_1;  // x_{t}
  x_3.tail(7) = x_2;  // x_{t+1}
  Eigen::VectorXd dx = (*f)(x_3);
  ASSERT_TRUE(dx.isApprox((x_2 - x_1) / .1));
}

TEST(cost_terms, finite_differences_acceleration) {
  std::srand(SEED);

  f = std::make_shared<FiniteDifferencesAcceleration>(1, .01);
  ASSERT_TRUE(f->CheckJacobian());
  ASSERT_TRUE(f->CheckHessian());

  f = std::make_shared<FiniteDifferencesAcceleration>(2, .01);
  ASSERT_TRUE(f->CheckJacobian(1e-6));

  f = std::make_shared<FiniteDifferencesAcceleration>(7, .1);
  ASSERT_TRUE(f->CheckJacobian());

  Eigen::VectorXd x_1 = Eigen::VectorXd::Random(7);
  Eigen::VectorXd x_2 = Eigen::VectorXd::Random(7);
  Eigen::VectorXd x_3 = Eigen::VectorXd::Random(7);
  Eigen::VectorXd x_4 = Eigen::VectorXd::Random(3 * 7);
  x_4.head(7) = x_1;        // x_{t-1}
  x_4.segment(7, 7) = x_2;  // x_{t}
  x_4.tail(7) = x_3;        // x_{t+1}
  Eigen::VectorXd dx = (*f)(x_4);
  ASSERT_TRUE(dx.isApprox((x_1  + x_3 - 2 * x_2) / (.1 * .1)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}