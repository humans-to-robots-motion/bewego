// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/motion/cost_terms.h>
#include <bewego/motion/trajectory.h>
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

TEST(trajectory, TrajectoryInterpolation) {
  uint32_t n = 10;
  uint32_t T = 30;
  auto q_init = util::Random(n);  // Sample hypercube.
  auto q_goal = util::Random(n);  // Sample hypercube.
  Trajectory trajectory = GetLinearInterpolation(q_init, q_goal, T);
  // auto matrix = trajectory.Matrix();
  // cout << "Trajectory : " << endl << matrix << endl;
  ASSERT_LT((q_init - trajectory.Configuration(0)).norm(), 1e-6);
  ASSERT_LT((q_goal - trajectory.FinalConfiguration()).norm(), 1e-6);
  ASSERT_LT((q_goal - trajectory.Configuration(T + 1)).norm(), 1e-6);
}

TEST(trajectory, TrajectoryNull) {
  uint32_t n = 10;
  uint32_t T = 30;
  auto q_init = util::Random(n);  // Sample hypercube.

  // Linear interpolation introduces numerical issues for testing
  // the trajectory NULL motion.
  Trajectory trajectory = GetLinearInterpolation(q_init, q_init, T);
  ASSERT_TRUE(IsTrajectoryNullMotion(trajectory, 1e-10));

  // Simple test
  for (uint32_t t = 1; t <= trajectory.T() + 1; t++) {
    trajectory.Configuration(t) = q_init;
  }
  ASSERT_TRUE(IsTrajectoryNullMotion(trajectory));
}

TEST(trajectory, ContinuousTrajectory) {
  uint32_t n = 10;
  uint32_t T = 30;
  auto q_init = util::Random(n);  // Sample hypercube.
  auto q_goal = util::Random(n);  // Sample hypercube.

  ContinuousTrajectory trajectory = GetLinearInterpolation(q_init, q_goal, T);

  // Test the length function
  double length_0 = (q_init - q_goal).norm();
  double length_1 = trajectory.length();
  // cout << "length_0 : " << length_0 << endl;
  // cout << "length_1 : " << length_1 << endl;
  ASSERT_LT(std::fabs(length_0 - length_1), 1.e-10);

  double ds = length_0 / T;
  for (uint32_t i = 0; i <= T; i++) {
    double alpha = float(i) / float(T);
    auto q_0 = trajectory.Configuration(i);
    auto q_1 = (1. - alpha) * q_init + alpha * q_goal;
    auto q_2 = trajectory.ConfigurationAtParameter(alpha);
    // cout << i << " -> (0) " << q_0.transpose() << endl;
    // cout << i << " -> (1) " << q_2.transpose() << endl;
    ASSERT_LT((q_0 - q_1).norm(), 1.e-7);
    ASSERT_LT((q_0 - q_2).norm(), 1.e-7);
  }
}

TEST(trajectory, Resample) {
  uint32_t n = 10;
  uint32_t T = 10;
  auto q_init = util::Random(n);  // Sample hypercube.
  auto q_goal = util::Random(n);  // Sample hypercube.
  auto t_0 = GetLinearInterpolation(q_init, q_goal, T);
  auto t_1 = Resample(t_0, 30);
  double l_0 = ContinuousTrajectory(t_0).length();
  double l_1 = ContinuousTrajectory(t_1).length();
  ASSERT_LT(std::fabs(l_0 - l_1), 1.e-7);
}

TEST(trajectory, ResampleTwoConfigs) {
  uint32_t n = 10;
  uint32_t T = 10;
  auto q_init = util::Random(n);  // Sample hypercube.
  auto q_goal = util::Random(n);  // Sample hypercube.
  std::vector<Eigen::VectorXd> configs{q_init, q_goal};
  auto t_0 = GetTrajectory(configs);
  auto t_1 = Resample(t_0, T);
  ASSERT_EQ(1, t_0.T());
  ASSERT_EQ(T, t_1.T());
  double l_0 = (q_init - q_goal).norm();
  double l_1 = ContinuousTrajectory(t_1).length();
  ASSERT_LT(std::fabs(l_0 - l_1), 1.e-7);
}

TEST(trajectory, ResampleTwoEqualConfigs) {
  uint32_t n = 10;
  uint32_t T = 10;
  auto q_init = util::Random(n);  // Sample hypercube.
  auto q_goal = q_init;
  std::vector<Eigen::VectorXd> configs{q_init, q_goal};
  auto t_0 = GetTrajectory(configs);
  auto t_1 = Resample(t_0, T);
  ASSERT_EQ(1, t_0.T());
  ASSERT_EQ(T, t_1.T());
  double l_0 = (q_init - q_goal).norm();
  double l_1 = ContinuousTrajectory(t_1).length();
  ASSERT_LT(std::fabs(l_0 - l_1), 1.e-7);
}
