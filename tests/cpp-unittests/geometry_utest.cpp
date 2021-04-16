// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/util/misc.h>
#include <bewego/workspace/geometry.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>

using namespace bewego;
using std::cout;
using std::endl;

const double error = 1e-05;
const double nb_test_points = 10;
const double resolution = 0.01;
static const uint32_t NB_TESTS = 10;
static const unsigned int SEED = 0;

TEST(geometry, easy) {
  double roll = 1.5707, pitch = 0, yaw = 0.707;
  Eigen::Vector3d rpy;
  rpy.x() = roll;
  rpy.y() = pitch;
  rpy.z() = yaw;
  Eigen::Vector4d q = EulerToQuaternion(rpy);
  Eigen::Quaterniond quaternion;
  quaternion = q;

  Eigen::Vector3d euler = quaternion.toRotationMatrix().eulerAngles(2, 1, 0);
  Eigen::Vector3d euler2;
  euler2.x() = euler[2];
  euler2.y() = euler[1];
  euler2.z() = euler[0];
  ASSERT_LT((rpy - euler2).norm(), 1e-3);
  cout << " - rpy   : " << rpy.transpose() << endl;
  cout << " - euler : " << euler.transpose() << endl;
}

TEST(geometry, urdf) {
  // this is the convention found in URDF dom

  Eigen::Vector3d rpy = Eigen::Vector3d::Random();
  double roll = rpy[0];
  double pitch = rpy[1];
  double yaw = rpy[2];

  double phi = roll / 2.0;
  double the = pitch / 2.0;
  double psi = yaw / 2.0;

  Eigen::Vector4d q1;
  q1.x() = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi);
  q1.y() = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi);
  q1.z() = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi);
  q1.w() = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi);
  q1.normalize();

  Eigen::Vector4d q2 = EulerToQuaternion(rpy);
  ASSERT_LT((q1 - q2).norm(), 1e-3);

  rpy[0] = -1.57079632679;
  rpy[1] = -1.57079632679;
  rpy[2] = 0;

  cout << " - matrix   : " << endl
       << Eigen::Quaterniond(EulerToQuaternion(rpy)).toRotationMatrix() << endl;
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
      function_tests_.push_back(std::make_pair(sphere, util::Random(dim)));
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
  RunAllTests();
  ASSERT_TRUE(function_tests_.front().first->type() == "SphereDistance");
}

class BoxDistanceTest : public DifferentialMapTest {
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
      auto rectangle = std::make_shared<BoxDistance>(center, dimension, theta);
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
      auto rectangle = std::make_shared<BoxDistance>(
          center, dimension, quaterion.toRotationMatrix());
      for (uint32_t i = 0; i < 5; ++i) {
        auto test = std::make_pair(rectangle, util::Random(dim));
        function_tests_.push_back(test);
      }
    }
  }
};

TEST_F(BoxDistanceTest, Evaluation) {
  set_verbose(false);
  set_precisions(error, 1e-3);
  // set_precisions(error, std::numeric_limits<double>::max());
  RunAllTests();
  ASSERT_TRUE(function_tests_.front().first->type() == "BoxDistance");
}

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
      auto r = std::make_shared<BoxDistance>(center, dimension, theta);
      auto p = Eigen::Rotation2Dd(theta) * query.first;
      EXPECT_NEAR(r->Evaluate(p), query.second, 1e-12);
    }
  }
}
