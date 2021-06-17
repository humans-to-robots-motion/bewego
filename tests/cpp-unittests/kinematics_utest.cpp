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

  // Add a single keypoint
  std::vector<std::pair<std::string, double>> keypoints;
  keypoints.push_back(std::make_pair("end", 0.1));

  auto robot =
      std::make_shared<Robot>(CreateThreeDofPlanarManipulator(), keypoints);

  // Test all maps for that keypoint
  for (auto n : std::vector<std::string>({"end", "end_x", "end_y", "end_z"})) {
    auto phi = robot->task_map(n);
    ASSERT_TRUE(phi->input_dimension() == 4);
    ASSERT_TRUE(phi->type() == "KinematicMap");
    AddRandomTests(phi, NB_TESTS);
  }
  RunAllTests();
}

TEST_F(DifferentialMapTest, robot_task_map_fixed) {
  std::srand(SEED);
  verbose_ = false;
  gradient_precision_ = 1e-6;

  // Add all keypoints
  std::vector<std::pair<std::string, double>> keypoints;
  for (auto n : std::vector<std::string>({"link1", "link2", "link3", "end"})) {
    keypoints.push_back(std::make_pair(n, 0.1));
  }

  bool with_fixed_end_link = true;
  auto kinematic_chain = CreateThreeDofPlanarManipulator(with_fixed_end_link);
  auto robot = std::make_shared<Robot>(kinematic_chain, keypoints);

  // Test position map for all keypoints
  for (uint32_t i = 0; i < robot->keypoints().size(); i++) {
    auto phi = robot->keypoint_map(i);
    ASSERT_TRUE(phi->input_dimension() == 3);
    ASSERT_TRUE(phi->type() == "KinematicMap");
    AddRandomTests(phi, NB_TESTS);
  }

  RunAllTests();
}
