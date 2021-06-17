// Copyright (c) 2021, University of Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/motion/hardcoded_robots.h>
#include <bewego/motion/robot.h>
#include <bewego/util/misc.h>

using namespace bewego;
using std::cout;
using std::endl;

static const uint32_t NB_TESTS = 10;
static const unsigned int SEED = 0;

TEST_F(DifferentialMapTest, robot_task_map) {
  std::srand(SEED);
  verbose_ = false;
  gradient_precision_ = 1e-6;

  std::vector<std::pair<std::string, double>> keypoints;
  keypoints.push_back(std::make_pair("link1", 0.1));
  keypoints.push_back(std::make_pair("link2", 0.1));
  keypoints.push_back(std::make_pair("link3", 0.1));
  keypoints.push_back(std::make_pair("end", 0.1));

  auto kinematic_chain = CreateThreeDofPlanarManipulator();
  auto robot = std::make_shared<Robot>(kinematic_chain, keypoints);

  auto phi0 = robot->task_map("end");
  auto phi1 = robot->task_map("end_x");
  auto phi2 = robot->task_map("end_y");
  auto phi3 = robot->task_map("end_z");

  AddRandomTests(phi0, NB_TESTS);
  AddRandomTests(phi1, NB_TESTS);
  AddRandomTests(phi2, NB_TESTS);
  AddRandomTests(phi3, NB_TESTS);

  RunAllTests();
  EXPECT_TRUE(phi0->type() == "KinematicMap");
}
