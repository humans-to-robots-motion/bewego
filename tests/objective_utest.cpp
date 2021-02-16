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

TEST(objective, motion_objective) {
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


TEST(objective, construct) {
  std::srand(SEED);
  double precision = 1e-6;
  uint32_t n = 2;
  double dt = .01;
  auto objective = std::make_shared<MotionObjective>(T, dt, n);

  // Workspace
  objective->AddSphere(Eigen::Vector2d(1, 1), .3);
  objective->AddSphere(Eigen::Vector2d(-1, -1), .3);
  objective->AddBox(Eigen::Vector2d(1, -1), Eigen::Vector2d(.1, .1));

  // Objectives
  objective->AddSmoothnessTerms(1, .1);
  objective->AddSmoothnessTerms(2, .1);
  objective->AddObstacleTerms(1.2, 10, 0);
  objective->AddWayPointTerms(Eigen::Vector2d(0, 0), 5, .7);
  objective->AddTerminalPotentialTerms(Eigen::Vector2d(-1, 1), .6);

  auto network = objective->function_network();
  ASSERT_TRUE(network->CheckJacobian(precision));
  ASSERT_TRUE(network->CheckHessian(precision));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}