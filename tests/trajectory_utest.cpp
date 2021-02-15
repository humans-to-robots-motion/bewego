// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/cost_terms.h>
#include <bewego/trajectory.h>
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

TEST(trajectory, cliques_function_network) {
  std::srand(SEED);

  double precision=1e-6;
  uint32_t T = 10;
  uint32_t n = 1;
  double dt = .01;

  auto f1 = std::make_shared<FiniteDifferencesVelocity>(n, dt);
  ASSERT_TRUE(f1->CheckJacobian());
  ASSERT_TRUE(f1->CheckHessian());

  auto network = std::make_shared<CliquesFunctionNetwork>((T + 2) * n, n);
  auto cost = std::make_shared<Compose>(f1, network->LeftOfCliqueMap());
  cost->set_debug(true);
  ASSERT_TRUE(cost->CheckJacobian(precision));
  ASSERT_TRUE(cost->CheckHessian(precision));

  // network->RegisterFunctionForAllCliques(cost);
  // ASSERT_TRUE(network->CheckJacobian(precision));
  // ASSERT_TRUE(network->CheckHessian(precision));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}