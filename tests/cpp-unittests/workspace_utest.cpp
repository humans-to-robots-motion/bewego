// Copyright (c) 2019, University of Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/util/misc.h>
#include <bewego/workspace/collision_checking.h>
#include <bewego/workspace/pixelmap.h>
#include <bewego/workspace/softmax_primitive_workspace.h>
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

std::shared_ptr<Workspace> CreateTest2DWorkspace() {
  std::vector<WorkspaceObjectPtr> objects;
  objects.push_back(std::make_shared<Circle>(Eigen::Vector2d(.5, .5), .2));
  objects.push_back(std::make_shared<Circle>(Eigen::Vector2d(0., 0.), .1));
  return std::make_shared<Workspace>(objects);
}

std::shared_ptr<Workspace> CreateTest3DWorkspace() {
  std::vector<WorkspaceObjectPtr> objects;
  objects.push_back(std::make_shared<Sphere>(Eigen::Vector3d(.5, .5, .5), .2));
  objects.push_back(std::make_shared<Box>(Eigen::Vector3d(.1, .1, .1),
                                          Eigen::Vector3d(.2, .2, .2),
                                          Eigen::Matrix3d::Identity()));
  return std::make_shared<Workspace>(objects);
}

TEST_F(DifferentialMapTest, smooth_collision_constraint) {
  std::srand(SEED);
  verbose_ = false;
  // gradient_precision_ = 1e-3;
  // hessian_precision_ = 1e-2;
  auto surfaces2d = CreateTest2DWorkspace()->ExtractSurfaceFunctions();
  auto phi2d = std::make_shared<SmoothCollisionConstraint>(surfaces2d, 10);
  AddRandomTests(phi2d, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi2d->type() == "SmoothCollisionConstraint");

  auto surfaces3d = CreateTest3DWorkspace()->ExtractSurfaceFunctions();
  auto phi3d = std::make_shared<SmoothCollisionConstraint>(surfaces3d, 10);
  AddRandomTests(phi3d, NB_TESTS);
  RunAllTests();
  ASSERT_TRUE(phi3d->type() == "SmoothCollisionConstraint");
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
  ASSERT_TRUE(phi->type() == "SoftDist");
}

TEST(cost_terms, obstacle_potential) {
  std::srand(SEED);
  uint32_t dim = 3;
  double dt = .01;
  double alpha = 10 * util::Rand();
  double scale = 10 * util::Rand();

  auto dist1 = std::make_shared<SquaredNorm>(dim);
  auto phi1 = std::make_shared<ObstaclePotential>(dist1, alpha, scale);
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    ASSERT_TRUE(phi1->CheckJacobian(1e-7));
    ASSERT_TRUE(phi1->CheckHessian(1e-7));
  }

  alpha = 10 * util::Rand();
  scale = 10 * util::Rand();
  Eigen::MatrixXd a = Eigen::MatrixXd::Random(dim, dim);
  Eigen::VectorXd b = Eigen::VectorXd::Random(dim);
  Eigen::VectorXd c = Eigen::VectorXd::Zero(1);
  auto dist2 = std::make_shared<QuadricMap>(a, b, c);
  auto phi2 = std::make_shared<ObstaclePotential>(dist2, alpha, scale);
  for (uint32_t i = 0; i < NB_TESTS; i++) {
    ASSERT_TRUE(phi2->CheckJacobian(1e-7));
    ASSERT_TRUE(phi2->CheckHessian(1e-7));
  }

  ASSERT_TRUE(dist1->type() == "SquaredNorm");
  ASSERT_TRUE(phi1->type() == "ObstaclePotential");
}

class LogisticActivationTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    function_tests_.clear();
    uint32_t dim = 3;
    uint32_t nb_tests = 10;

    Eigen::VectorXd p0 = .5 * Eigen::VectorXd::Ones(dim);
    auto phi = std::make_shared<SphereDistance>(p0, .5);
    auto f1 = LogisticActivation(phi, 10);
    // auto f2 = Shift(Negate(LogisticActivation(phi, 10)), 1);
    for (uint32_t i = 0; i < nb_tests; ++i) {
      Eigen::VectorXd x = util::Random(dim);
      function_tests_.push_back(std::make_pair(f1, x));
      // function_tests_.push_back(std::make_pair(f2, x));
    }
  }
};

TEST_F(LogisticActivationTest, Evaluation) {
  set_verbose(false);
  set_precisions(1e-6, 1e-5);
  RunAllTests();
}
