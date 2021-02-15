// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/cost_terms.h>
#include <bewego/objective.h>
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
static const uint32_t T = 10;

std::shared_ptr<CliquesFunctionNetwork> MakeFunctionNetwork(uint32_t n = 1) {
  return std::make_shared<CliquesFunctionNetwork>((T + 2) * n, n);
}

TEST(cliques_function_network, cliques) {
  std::srand(SEED);
  double precision = 1e-6;
  uint32_t n = 0;
  std::shared_ptr<CliquesFunctionNetwork> network;
  std::vector<Eigen::VectorXd> cliques;
  Eigen::VectorXd x;

  // -------- C-Space dimension 1

  n = 1;
  network = MakeFunctionNetwork(n);
  x = Eigen::VectorXd::Random(network->input_dimension());
  EXPECT_EQ(network->nb_cliques(), 10);
  cliques = network->AllCliques(x);
  EXPECT_EQ(cliques.size(), network->nb_cliques());
  EXPECT_EQ(cliques.size(), T);
  for (uint32_t t = 0; t < T; t++) {
    EXPECT_NEAR(cliques[t][0], x[t], 1e-6);
  }

  // -------- C-Space dimension 7

  n = 7;
  network = MakeFunctionNetwork(n);
  x = Eigen::VectorXd::Random(network->input_dimension());
  EXPECT_EQ(network->nb_cliques(), 10);
  cliques = network->AllCliques(x);
  EXPECT_EQ(cliques.size(), network->nb_cliques());
  EXPECT_EQ(cliques.size(), T);
  for (uint32_t t = 0; t < T; t++) {
    for (uint32_t i = 0; i < n; i++) {
      EXPECT_NEAR(cliques[t][i], x[t * n + i], 1e-6);
    }
  }
}

TEST(cliques_function_network, jacobian) {
  std::srand(SEED);

  double precision = 1e-6;
  uint32_t n = 2;
  double dt = .01;

  DifferentiableMapPtr f1, cost;
  std::shared_ptr<CliquesFunctionNetwork> network;

  // --------

  f1 = std::make_shared<SquaredNormVelocity>(n, dt);
  ASSERT_TRUE(f1->CheckJacobian(precision));
  ASSERT_TRUE(f1->CheckHessian(precision));

  network = MakeFunctionNetwork(n);

  cost = std::make_shared<Compose>(f1, network->LeftOfCliqueMap());
  ASSERT_TRUE(cost->CheckJacobian(precision));
  ASSERT_TRUE(cost->CheckHessian(precision));

  network->RegisterFunctionForAllCliques(cost);
  ASSERT_TRUE(network->CheckJacobian(precision));
  ASSERT_TRUE(network->CheckHessian(precision));

  // --------

  network = MakeFunctionNetwork(n);

  cost = std::make_shared<SquaredNormAcceleration>(n, dt);
  ASSERT_TRUE(cost->CheckJacobian(precision));
  ASSERT_TRUE(cost->CheckHessian(precision));

  network->RegisterFunctionForAllCliques(cost);
  ASSERT_TRUE(network->CheckJacobian(precision));
  ASSERT_TRUE(network->CheckHessian(precision));
}

TEST(cliques_function_network, motion_objective) {
  std::srand(SEED);
  double precision = 1e-6;
  uint32_t n = 2;
  double dt = .01;
  auto objective = std::make_shared<MotionObjective>(T, dt, n);
  objective->AddSmoothnessTerms(1, .1);
  objective->AddSmoothnessTerms(2, .1);
  auto network = objective->function_network();
  ASSERT_TRUE(network->CheckJacobian(precision));
  ASSERT_TRUE(network->CheckHessian(precision));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}