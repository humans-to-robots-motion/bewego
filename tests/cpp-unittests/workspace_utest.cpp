// Copyright (c) 2019, University of Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/util/misc.h>
#include <bewego/workspace/pixelmap.h>
#include <bewego/workspace/workspace.h>
#include <gtest/gtest.h>

using namespace bewego;
using std::cout;
using std::endl;

const double error = 1e-05;
const double nb_test_points = 10;
const double resolution = 0.01;
static const uint32_t NB_TESTS = 10;
static const unsigned int SEED = 0;

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
        Eigen::VectorXd p = bewego::util::Random(dim);
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
  RunAllTests();
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
      auto r = std::make_shared<RectangleDistance>(center, dimension, theta);
      auto p = Eigen::Rotation2Dd(theta) * query.first;
      EXPECT_NEAR(r->Evaluate(p), query.second, 1e-12);
    }
  }
}

std::shared_ptr<Workspace> CreateTestWorkspace() {
  std::vector<WorkspaceObjectPtr> objects;
  objects.push_back(std::make_shared<Circle>(Eigen::Vector2d(.5, .5), .2));
  objects.push_back(std::make_shared<Circle>(Eigen::Vector2d(0., 0.), .1));
  return std::make_shared<Workspace>(objects);
}

TEST_F(DifferentialMapTest, smooth_collision_constraints) {
  std::srand(SEED);
  verbose_ = false;
  // gradient_precision_ = 1e-3;
  // hessian_precision_ = 1e-2;
  auto surfaces = CreateTestWorkspace()->ExtractSurfaceFunctions();
  auto phi = std::make_shared<SmoothCollisionConstraints>(surfaces, 10);
  AddRandomTests(phi, NB_TESTS);
  RunAllTests();
}

TEST_F(DifferentialMapTest, soft_sphere_distance) {
  std::srand(SEED);
  // verbose_ = true;
  // set_precisions(1e-6, 1e-5);

  double radius = .10;
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);
  auto dist_sq = ComposedWith(std::make_shared<SquareMap>(),
                              std::make_shared<SphereDistance>(x0, radius));
  auto phi = std::make_shared<SoftDist>(dist_sq);
  AddRandomTests(phi, NB_TESTS);
  RunAllTests();
}
