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

TEST(cliques_function_network, cliques_accessors) {
  std::srand(SEED);
  double precision = 1e-6;
  uint32_t n = 0;
  std::shared_ptr<CliquesFunctionNetwork> network;
  std::vector<Eigen::VectorXd> cliques;
  Eigen::VectorXd x;

  // -------- C-Space dimension 7

  n = 7;
  network = MakeFunctionNetwork(n);

  x = Eigen::VectorXd::Random(network->input_dimension());
  EXPECT_EQ(network->nb_cliques(), 10);
  cliques = network->AllCliques(x);
  EXPECT_EQ(cliques.size(), network->nb_cliques());
  EXPECT_EQ(cliques.size(), T);

  auto f1 = network->CenterOfCliqueMap();     // x_{t}
  auto f2 = network->RightMostOfCliqueMap();  // x_{t+1}
  auto f3 = network->RightOfCliqueMap();      // x_{t} ; x_{t+1}
  auto f4 = network->LeftMostOfCliqueMap();   // x_{t-1}
  auto f5 = network->LeftOfCliqueMap();       // x_{t-1} ; x_{t}

  for (uint32_t t = 0; t < 4; t++) {
    Eigen::VectorXd x1 = cliques[t].segment(n, n);      // x_{t}
    Eigen::VectorXd x2 = cliques[t].segment(2 * n, n);  // x_{t+1}
    Eigen::VectorXd x3 = cliques[t].segment(0, n);      // x_{t} ; x_{t+1}
    Eigen::VectorXd x4 = cliques[t].segment(n, 2 * n);  // x_{t-1}
    Eigen::VectorXd x5 = cliques[t].segment(0, 2 * n);  // x_{t-1} ; x_{t}

    EXPECT_LT((x1 - (*f1)(cliques[t])).norm(), 1e-6);
    EXPECT_LT((x2 - (*f2)(cliques[t])).norm(), 1e-6);
    EXPECT_LT((x3 - (*f4)(cliques[t])).norm(), 1e-6);
    EXPECT_LT((x4 - (*f3)(cliques[t])).norm(), 1e-6);
    EXPECT_LT((x5 - (*f5)(cliques[t])).norm(), 1e-6);
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}