// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/motion/freeflyers.h>
#include <bewego/util/misc.h>
#include <bewego/workspace/softmax_primitive_workspace.h>
#include <bewego/workspace/workspace.h>

using namespace bewego;
using std::cout;
using std::endl;

const double alpha = 20;
static const uint32_t NB_TESTS = 10;
static const unsigned int SEED = 0;

std::shared_ptr<const Workspace> CreateTestCircleWorkspace() {
  std::vector<std::shared_ptr<const WorkspaceObject>> circles = {
      std::make_shared<Circle>(Eigen::Vector2d(.2, .5), .1),
      std::make_shared<Circle>(Eigen::Vector2d(.1, .4), .1)};
  return std::make_shared<WorkspacePotentalPrimitive>(circles, alpha);
}

std::shared_ptr<const Workspace> CreateTestSphereWorkspace() {
  std::vector<std::shared_ptr<const WorkspaceObject>> spheres = {
      std::make_shared<Sphere>(Eigen::Vector3d(.2, .5, 0), .1),
      std::make_shared<Sphere>(Eigen::Vector3d(.1, .4, 0), .1)};
  return std::make_shared<WorkspacePotentalPrimitive>(spheres, alpha);
}

TEST_F(DifferentialMapTest, freeflyer_task_map) {
  std::srand(SEED);
  set_verbose(false);
  // Here the hessian is approximated by pullback
  // so we don't expect a tight precision.
  set_precisions(1e-6, 1e-3);
  RunAllTests();
  auto freeflyer = MakeFreeflyer2D();
  auto phi1 = freeflyer->keypoint_map(0);
  auto phi2 = freeflyer->keypoint_map(1);
  AddRandomTests(phi1, NB_TESTS);
  AddRandomTests(phi2, NB_TESTS);
  RunAllTests();
  EXPECT_TRUE(phi1->type() == "HomogeneousTransform2d");
  EXPECT_TRUE(phi2->type() == "HomogeneousTransform2d");
}

class FreeFlyerCollisionConstraintsTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    // 2D workspace
    workspace_ = CreateTestCircleWorkspace();
    freeflyer_ = MakeFreeflyer2D();
    collision_checker_ = std::make_shared<FreeFlyerCollisionConstraints>(
        freeflyer_, workspace_->ExtractSurfaceFunctions());
    constraint_ = collision_checker_->smooth_constraint();
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = util::Random(constraint_->input_dimension());
      function_tests_.push_back(std::make_pair(constraint_, x));
    }

    // 3D workspace
    workspace_ = CreateTestSphereWorkspace();
    freeflyer_ = MakeFreeflyer3D();
    collision_checker_ = std::make_shared<FreeFlyerCollisionConstraints>(
        freeflyer_, workspace_->ExtractSurfaceFunctions());
    constraint_ = collision_checker_->smooth_constraint();
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = util::Random(constraint_->input_dimension());
      function_tests_.push_back(std::make_pair(constraint_, x));
    }
  }

  std::shared_ptr<const Workspace> workspace_;
  std::string description_file_;
  std::shared_ptr<const Freeflyer> freeflyer_;
  std::shared_ptr<const FreeFlyerCollisionConstraints> collision_checker_;
  std::shared_ptr<const DifferentiableMap> constraint_;
};

/** TODO */
TEST_F(FreeFlyerCollisionConstraintsTest, Evaluation) {
  set_verbose(false);
  // Here the hessian is approximated by pullback
  // so we don't expect a tight precision.
  set_precisions(1e-6, 1e-3);
  RunAllTests();
}
