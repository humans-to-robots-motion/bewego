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

class FreeFlyerCollisionConstraintsTest : public DifferentialMapTest {
 public:
  virtual void SetUp() {
    // 2D workspace
    workspace_ = CreateTestCircleWorkspace();
    // description_file_ =
    //     ros::package::getPath("rieef_utils") + "/config/freeflyer_2d.json";
    // freeflyer_ = LoadFreeFlyerFromJsonFile(description_file_);
    collision_checker_ = std::make_shared<FreeFlyerCollisionConstraints>(
        freeflyer_, workspace_->ExtractSurfaceFunctions());
    constraint_ = collision_checker_->smooth_constraint();
    for (uint32_t i = 0; i < 10; ++i) {
      Eigen::VectorXd x = util::Random(constraint_->input_dimension());
      function_tests_.push_back(std::make_pair(constraint_, x));
    }

    // 3D workspace
    workspace_ = CreateTestSphereWorkspace();
    // description_file_ =
    //     ros::package::getPath("rieef_utils") + "/config/freeflyer_3d.json";
    // freeflyer_ = LoadFreeFlyerFromJsonFile(description_file_);
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

/** TODO test freeflyer constraint !!!
TEST_F(FreeFlyerCollisionConstraintsTest, Evaluation) {
  set_verbose(false);
  // Here the hessian is approximated by pullback
  // so we don't expect a tight precision.
  set_precisions(1e-6, 1e3);
  RunAllTests();
}
*/
