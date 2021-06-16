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
  set_verbose(false);
  // Here the hessian is approximated by pullback
  // so we don't expect a tight precision.
  set_precisions(1e-6, 1e-3);

  auto kinematic_chain = CreateThreeDofPlanarManipulator();
  std::vector<std::pair<std::string, double>> keypoints;
  auto robot = std::make_shared<Robot>(kinematic_chain, keypoints);

  //  freeflyer->keypoint_map(0);
  // auto phi2 = freeflyer->keypoint_map(1);
  // AddRandomTests(phi1, NB_TESTS);
  // AddRandomTests(phi2, NB_TESTS);
  // RunAllTests();
  // EXPECT_TRUE(phi1->type() == "HomogeneousTransform2d");
  // EXPECT_TRUE(phi2->type() == "HomogeneousTransform2d");
}
