// Copyright (c) 2020, Univeristy of Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/geodesic_flow/attractors.h>
#include <bewego/util/misc.h>
#include <gtest/gtest.h>

#include <iostream>

using namespace bewego;
using namespace std;
using std::cout;
using std::endl;

TEST(SmoothAttractor, test) {
  uint32_t dim = 3;
  double d_transition = 1.;

  // 1) Weird test quadric function for the goal potential
  Eigen::MatrixXd H1;
  Eigen::VectorXd b1;
  double c1;
  H1 = Eigen::MatrixXd::Random(dim, dim);
  H1 = H1 * H1.transpose();
  b1 = util::Random(dim);
  c1 = util::Rand();
  auto d = std::make_shared<QuadricMap>(H1, b1, c1);

  // 2) create a smooth attractor around a random goal
  Eigen::VectorXd x_goal = util::Random(dim);
  auto f = std::make_shared<SmoothAttractor>(d, x_goal, d_transition, .3);
  cout << "k : " << f->k() << endl;
  cout << "x_goal : " << x_goal.transpose() << endl;

  double v1 = (*f->smooth_distance())(x_goal)[0];
  cout << " - v1 : " << v1 << endl;
  ASSERT_LT(std::abs(v1), 1e-7);

  // Check that it's 0 at goal
  double v2 = (*f)(x_goal)[0];
  cout << " - v2 : " << v2 << endl;
  cout << " - v3 : " << (*f->distance())(x_goal)[0] << endl;
  ASSERT_LT(std::abs(v2), 1e-7);

  double e1, e2;
  for (uint32_t i = 0; i < 100; i++) {
    Eigen::VectorXd x_r = Eigen::VectorXd::Random(dim);

    // Test point far from goal.
    Eigen::VectorXd x1 = 1e+3 * x_r + x_goal;
    double f_1 = (*f)(x1)[0];
    e1 = std::abs(f_1 - (*f->smooth_distance())(x1)[0]);
    e2 = std::abs(f_1 - (*f->distance())(x1)[0]);
    ASSERT_GE(e1, e2);  // TODO this should be ASSERT_GT

    // Test point close to goal.
    Eigen::VectorXd x2 = 1e-3 * x_r + x_goal;
    double f_2 = (*f)(x2)[0];
    e1 = std::abs(f_2 - (*f->smooth_distance())(x2)[0]);
    e2 = std::abs(f_2 - (*f->distance())(x2)[0]);
    ASSERT_LE(e1, e2);  // TODO this should be ASSERT_LT
  }
}

TEST(Attractors, workspace) {
  double diff = 0;
  EXPECT_NEAR(diff, 0., 1e-2);
}
