// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/atomic_operators.h>
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

TEST(atomic_operators, zero_map) {
  std::srand(SEED);

  f = std::make_shared<ZeroMap>(5, 5);
  ASSERT_TRUE(f->CheckJacobian());

  f = std::make_shared<ZeroMap>(1, 5);
  ASSERT_TRUE(f->CheckHessian());
}

TEST(atomic_operators, identity_map) {
  std::srand(SEED);

  f = std::make_shared<IdentityMap>(5);
  ASSERT_TRUE(f->CheckJacobian());

  f = std::make_shared<IdentityMap>(1);
  ASSERT_TRUE(f->CheckHessian());
}

TEST(atomic_operators, affine_map) {
  std::srand(SEED);
  const double precision = 1e-10;
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Random(3);
    f = std::make_shared<AffineMap>(a, b);
    ASSERT_TRUE(f->CheckJacobian(precision));

    a = Eigen::MatrixXd::Random(1, 2);
    b = Eigen::VectorXd::Random(1);
    f = std::make_shared<AffineMap>(a, b);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }
}

TEST(atomic_operators, squared_norm) {
  std::srand(SEED);
  const double precision = 1e-10;
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    Eigen::VectorXd x0 = Eigen::VectorXd::Random(2);
    f = std::make_shared<SquaredNorm>(x0);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }
}

TEST(atomic_operators, range_subspace_map) {
  std::srand(SEED);
  const double precision = 1e-10;
  uint32_t n = 10;
  std::vector<uint32_t> indices;

  indices = {2, 3, 7};  // output_dimension = 3
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    f = std::make_shared<RangeSubspaceMap>(n, indices);
    ASSERT_TRUE(f->CheckJacobian(precision));
  }

  indices = {5};  // output_dimension = 1
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    f = std::make_shared<RangeSubspaceMap>(n, indices);
    ASSERT_TRUE(f->CheckJacobian(precision));
    ASSERT_TRUE(f->CheckHessian(precision));
  }
}

TEST(atomic_operators, sum_map) {
  std::srand(SEED);
  const double precision = 1e-10;
  auto maps = std::make_shared<VectorOfMaps>();
  for (uint32_t i = 0; i < 3; i++) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 2);
    Eigen::VectorXd b = Eigen::VectorXd::Random(3);
    maps->push_back(std::make_shared<AffineMap>(a, b));
  }

  f = std::make_shared<SumMap>(maps);
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    ASSERT_TRUE(f->CheckJacobian(precision));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}