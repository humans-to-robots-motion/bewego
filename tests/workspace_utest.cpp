// Copyright (c) 2019, University of Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/util.h>
#include <bewego/workspace.h>
#include <gtest/gtest.h>

using namespace bewego;
using std::cout;
using std::endl;

const double error = 1e-05;
const double nb_test_points = 10;

TEST(RectangleDistance, Main) {
  Eigen::Vector2d center(0, 0);
  Eigen::Vector2d dimension(.1, .1);  // half dim
  double d = std::sqrt(2. * .1 * .1);

  std::vector<std::pair<Eigen::Vector2d, double>> points_dist;
  points_dist.push_back(std::make_pair(Eigen::Vector2d(.2, 0.), .1));
  points_dist.push_back(std::make_pair(Eigen::Vector2d(.0, .2), .1));
  points_dist.push_back(std::make_pair(Eigen::Vector2d(.05, 0.), -.05));
  points_dist.push_back(std::make_pair(Eigen::Vector2d(0., .05), -.05));
  points_dist.push_back(std::make_pair(Eigen::Vector2d(.05, 0.01), -.05));
  points_dist.push_back(std::make_pair(Eigen::Vector2d(0.01, .05), -.05));
  points_dist.push_back(std::make_pair(Eigen::Vector2d(.2, .2), d));
  points_dist.push_back(std::make_pair(Eigen::Vector2d(-.2, -.2), d));

  for (uint32_t i = 0; i < 20; i++) {
    for (const auto& query : points_dist) {
      double theta = 2 * M_PI * util::Rand();
      auto r = std::make_shared<RectangleDistance>(center, dimension, theta);
      auto p = Eigen::Rotation2Dd(theta) * query.first;
      EXPECT_NEAR(r->Evaluate(p), query.second, 1e-12);
    }
  }
}

class RectangleDistanceTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    std::srand(1);
    function_tests_.clear();
    Add2DPoints();
    Add3DPoints();
  }

  void Add2DPoints() {
    uint32_t dim = 2;
    Eigen::Vector2d center(.5, .5);
    Eigen::Vector2d dimension(.1, .1);
    for (double theta = 0; theta < 3.14; theta += .1) {
      auto rectangle =
          std::make_shared<RectangleDistance>(center, dimension, theta);
      for (uint32_t i = 0; i < 5; ++i) {
        Eigen::VectorXd p = util::Random(dim);
        auto test = std::make_pair(rectangle, p);
        function_tests_.push_back(test);
      }
    }
  }

  void Add3DPoints() {
    uint32_t dim = 3;
    Eigen::Vector3d center(.5, .5, .5);
    Eigen::Vector3d dimension(.1, .1, .1);
    for (double theta = 0; theta < 3.14; theta += .1) {
      Eigen::Quaterniond quaterion =
          Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX());
      auto rectangle = std::make_shared<RectangleDistance>(
          center, dimension, quaterion.toRotationMatrix());
      for (uint32_t i = 0; i < 5; ++i) {
        auto test = std::make_pair(rectangle, util::Random(dim));
        function_tests_.push_back(test);
      }
    }
  }
};

TEST_F(RectangleDistanceTest, Evaluation) {
  set_verbose(false);
  set_precisions(error, 1e-3);
  // set_precisions(error, std::numeric_limits<double>::max());
  RunTests();
}

class SphereDistanceTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    std::srand(1);
    function_tests_.clear();
    Add2DPoints();
    Add3DPoints();
  }

  void Add2DPoints() {
    uint32_t dim = 2;
    Eigen::Vector2d center(.5, .5);
    double radius(.1);
    auto sphere = std::make_shared<SphereDistance>(center, radius);
    for (uint32_t i = 0; i < 5; ++i) {
      Eigen::VectorXd p = util::Random(dim);
      function_tests_.push_back(std::make_pair(sphere, p));
    }
  }

  void Add3DPoints() {
    uint32_t dim = 3;
    Eigen::Vector3d center(.5, .5, .5);
    double radius(.1);
    auto sphere = std::make_shared<SphereDistance>(center, radius);
    for (uint32_t i = 0; i < 5; ++i) {
      function_tests_.push_back(std::make_pair(sphere, util::Random(dim)));
    }
  }
};

TEST_F(SphereDistanceTest, Evaluation) {
  set_verbose(false);
  set_precisions(error, 1e-3);
  // set_precisions(error, std::numeric_limits<double>::max());
  RunTests();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
