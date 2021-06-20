/*
 * Copyright (c) 2019
 * All rights reserved.
 *
 * Redistribution  and  use  in  source  and binary  forms,  with  or  without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of  source  code must retain the  above copyright
 *      notice and this list of conditions.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice and  this list of  conditions in the  documentation and/or
 *      other materials provided with the distribution.
 *
 * THE SOFTWARE  IS PROVIDED "AS IS"  AND THE AUTHOR  DISCLAIMS ALL WARRANTIES
 * WITH  REGARD   TO  THIS  SOFTWARE  INCLUDING  ALL   IMPLIED  WARRANTIES  OF
 * MERCHANTABILITY AND  FITNESS.  IN NO EVENT  SHALL THE AUTHOR  BE LIABLE FOR
 * ANY  SPECIAL, DIRECT,  INDIRECT, OR  CONSEQUENTIAL DAMAGES  OR  ANY DAMAGES
 * WHATSOEVER  RESULTING FROM  LOSS OF  USE, DATA  OR PROFITS,  WHETHER  IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR  OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 *
 *                                                             Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/motion/forward_kinematics.h>
#include <bewego/motion/freeflyers.h>
#include <bewego/motion/hardcoded_robots.h>
#include <bewego/motion/objective.h>
#include <bewego/motion/robot.h>
#include <bewego/motion/trajectory.h>
#include <bewego/planning/planar_grid.h>
#include <bewego/planning/value_iteration.h>
#include <bewego/util/interpolation.h>
#include <bewego/workspace/geometry.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef WITH_IPOPT
#include <bewego/numerical_optimization/bewopt_freeflyer.h>
#include <bewego/numerical_optimization/bewopt_planar.h>
#include <bewego/numerical_optimization/bewopt_robot.h>
#endif

#include <iostream>
using std::cout;
using std::endl;

int add(int i, int j) { return i + j; }

bool test_identity(int n) {
  uint32_t dimension = n;
  auto f = std::make_shared<bewego::IdentityMap>(dimension);
  auto x = Eigen::VectorXd::Random(dimension);
  Eigen::MatrixXd J1, J2;
  J1 = f->Jacobian(x);
  J2 = bewego::DifferentiableMap::FiniteDifferenceJacobian(*f, x);
  bool a = J1.isApprox(J2);
  bool b = true;
  if (dimension == 1) {
    Eigen::MatrixXd H1, H2;
    H1 = f->Hessian(x);
    H2 = bewego::DifferentiableMap::FiniteDifferenceHessian(*f, x);
    b = H1.isApprox(H2);
  }
  return a && b;
}

namespace py = pybind11;
namespace bewego {

//! Gives access to some basic functions
class AStarGrid {
 public:
  AStarGrid() { problem_ = std::make_shared<AStarProblem>(); }

  // Getters.
  double pace() const { return problem_->pace(); }
  std::vector<double> env_size() const { return problem_->env_size(); }

  void init_grid(double p, const std::vector<double>& e) {
    problem_->set_pace(p);
    problem_->set_env_size(e);
    problem_->InitGrid();
  }
  void set_costs(const Eigen::MatrixXd& costs) { problem_->InitCosts(costs); }
  bool solve(const Eigen::Vector2i& s, const Eigen::Vector2i& g) {
    return problem_->Solve(s, g);
  }
  Eigen::MatrixXi path() const { return problem_->PathCoordinates(); }

 protected:
  std::shared_ptr<AStarProblem> problem_;
};

}  // namespace bewego

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

PYBIND11_MODULE(_pybewego, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: bewego
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

  m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

  m.def(
      "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

  m.def("test_identity", &test_identity, R"pbdoc(
        Test the identity map
        Some other explanation about the subtract function.
    )pbdoc");

  m.def("quaternion_to_matrix", &bewego::QuaternionToMatrix, R"pbdoc(
        Returns the rotation matrix corresponding to a Quaternion
    )pbdoc");

  m.def("euler_to_quaternion", &bewego::EulerToQuaternion, R"pbdoc(
        Returns the quaternion corresponding to euler
    )pbdoc");

  m.def("create_freeflyer", &bewego::CreateFreeFlyer, R"pbdoc(
        Returns a Freeflyer
    )pbdoc");

  m.def("create_planar_robot_kinematics",
        &bewego::CreateThreeDofPlanarManipulator, py::arg("last_fixed") = false,
        R"pbdoc(
        Returns a Planar robot
    )pbdoc");

  py::class_<bewego::AStarGrid>(m, "AStarGrid")
      .def(py::init<>())
      .def("pace", &bewego::AStarGrid::pace)
      .def("env_size", &bewego::AStarGrid::env_size)
      .def("init_grid", &bewego::AStarGrid::init_grid)
      .def("set_costs", &bewego::AStarGrid::set_costs)
      .def("solve", &bewego::AStarGrid::solve)
      .def("path", &bewego::AStarGrid::path);

  py::class_<bewego::ValueIteration>(m, "ValueIteration")
      .def(py::init<>())
      .def("set_max_iterations", &bewego::ValueIteration::set_max_iterations)
      .def("set_theta", &bewego::ValueIteration::set_theta)
      .def("run", &bewego::ValueIteration::Run)
      .def("solve", &bewego::ValueIteration::solve);

  py::class_<bewego::LWR, std::shared_ptr<bewego::LWR>>(m, "LWR")
      .def(py::init<uint32_t, uint32_t>())
      .def("input_dimension", &bewego::LWR::input_dimension)
      .def("output_dimension", &bewego::LWR::output_dimension)
      .def("gradient", &bewego::LWR::Gradient)
      .def("forward", &bewego::LWR::Forward)
      .def("jacobian", &bewego::LWR::Jacobian)
      .def("hessian", &bewego::LWR::Hessian)
      .def("multi_forward", &bewego::LWR::ForwardMultiQuerry)
      .def("multi_jacobian", &bewego::LWR::JacobianMultiQuerry)
      .def("initialize", &bewego::LWR::Initialize)
      .def_readwrite("X", &bewego::LWR::X_)
      .def_readwrite("Y", &bewego::LWR::Y_)
      .def_readwrite("D", &bewego::LWR::D_)
      .def_readwrite("ridge_lambda", &bewego::LWR::ridge_lambda_)
      .def("__call__", &bewego::LWR::Forward, py::arg("e") = nullptr,
           py::is_operator());

  py::class_<bewego::DifferentiableMap,
             std::shared_ptr<bewego::DifferentiableMap>>(m, "DifferentiableMap")
      .def("input_dimension", &bewego::DifferentiableMap::input_dimension)
      .def("output_dimension", &bewego::DifferentiableMap::output_dimension)
      .def("gradient", &bewego::DifferentiableMap::Gradient)
      .def("forward", &bewego::DifferentiableMap::Forward)
      .def("jacobian", &bewego::DifferentiableMap::Jacobian)
      .def("hessian", &bewego::DifferentiableMap::Hessian)
      .def("__call__", &bewego::DifferentiableMap::Forward,
           py::arg("e") = nullptr, py::is_operator());

  py::class_<bewego::ObstaclePotential,
             std::shared_ptr<bewego::ObstaclePotential>>(m, "ObstaclePotential")
      .def("input_dimension", &bewego::ObstaclePotential::input_dimension)
      .def("output_dimension", &bewego::ObstaclePotential::output_dimension)
      .def("gradient", &bewego::ObstaclePotential::Gradient)
      .def("forward", &bewego::ObstaclePotential::Forward)
      .def("jacobian", &bewego::ObstaclePotential::Jacobian)
      .def("hessian", &bewego::ObstaclePotential::Hessian)
      .def("__call__", &bewego::ObstaclePotential::Forward,
           py::arg("e") = nullptr, py::is_operator());

  py::class_<bewego::Freeflyer, std::shared_ptr<bewego::Freeflyer>>(
      m, "Freeflyer2D")
      .def("keypoint_map", &bewego::Freeflyer::keypoint_map)
      .def("keypoint_radius", &bewego::Freeflyer::keypoint_radius)
      .def("n", &bewego::Freeflyer::n);

  py::class_<bewego::RigidBodyInfo>(m, "RigidBodyInfo")
      .def(py::init<>())
      .def_readwrite("name", &bewego::RigidBodyInfo::name)
      .def_readwrite("joint_name", &bewego::RigidBodyInfo::joint_name)
      .def_readwrite("joint_type", &bewego::RigidBodyInfo::joint_type)
      .def_readwrite("dof_lower_limit", &bewego::RigidBodyInfo::dof_lower_limit)
      .def_readwrite("dof_upper_limit", &bewego::RigidBodyInfo::dof_upper_limit)
      .def_readwrite("local_in_prev", &bewego::RigidBodyInfo::local_in_prev)
      .def_readwrite("joint_axis_in_local",
                     &bewego::RigidBodyInfo::joint_axis_in_local);

  py::class_<bewego::KinematicChain, std::shared_ptr<bewego::KinematicChain>>(
      m, "KinematicChain")
      .def(py::init<>())
      .def("add_rigid_body", &bewego::KinematicChain::AddRigidBody)
      .def("set_base_transform", &bewego::KinematicChain::set_base_transform)
      .def("set_and_update", &bewego::KinematicChain::SetAndUpdate)
      .def("position", &bewego::KinematicChain::position)
      .def("rotation", &bewego::KinematicChain::rotation)
      .def("transform", &bewego::KinematicChain::transform)
      .def("jacobian_pos", &bewego::KinematicChain::JacobianPosition)
      .def("jacobian_axis", &bewego::KinematicChain::JacobianAxis)
      .def("jacobian_frame", &bewego::KinematicChain::JacobianFrame);

  py::class_<bewego::KinematicMap, std::shared_ptr<bewego::KinematicMap>>(
      m, "KinematicMap")
      .def("input_dimension", &bewego::KinematicMap::input_dimension)
      .def("output_dimension", &bewego::KinematicMap::output_dimension)
      .def("name", &bewego::KinematicMap::name)
      .def("forward", &bewego::KinematicMap::Forward)
      .def("jacobian", &bewego::KinematicMap::Jacobian)
      .def("__call__", &bewego::KinematicMap::Forward, py::arg("e") = nullptr,
           py::is_operator());

  py::class_<bewego::Robot, std::shared_ptr<bewego::Robot>>(m, "Robot")
      .def(py::init<const std::vector<bewego::RigidBodyInfo>&,
                    const std::vector<std::pair<std::string, double>>&>())
      .def("task_map", &bewego::Robot::task_map)
      .def("keypoint_map", &bewego::Robot::keypoint_map)
      .def("keypoint_radius", &bewego::Robot::keypoint_radius);

  py::class_<bewego::TrajectoryObjectiveFunction,
             std::shared_ptr<bewego::TrajectoryObjectiveFunction>>(
      m, "TrajectoryObjectiveFunction")
      .def("input_dimension",
           &bewego::TrajectoryObjectiveFunction::input_dimension)
      .def("output_dimension",
           &bewego::TrajectoryObjectiveFunction::output_dimension)
      .def("gradient", &bewego::TrajectoryObjectiveFunction::Gradient)
      .def("forward", &bewego::TrajectoryObjectiveFunction::Forward)
      .def("jacobian", &bewego::TrajectoryObjectiveFunction::Jacobian)
      .def("hessian", &bewego::TrajectoryObjectiveFunction::Hessian)
      .def("__call__", &bewego::TrajectoryObjectiveFunction::Forward,
           py::arg("e") = nullptr, py::is_operator());

  py::class_<bewego::MotionObjective>(m, "MotionObjective")
      .def(py::init<uint32_t, double, uint32_t>())
      .def("add_smoothness_terms", &bewego::MotionObjective::AddSmoothnessTerms)
      .def("add_obstacle_terms", &bewego::MotionObjective::AddObstacleTerms)
      .def("add_terminal_potential_terms",
           &bewego::MotionObjective::AddTerminalPotentialTerms)
      .def("add_waypoint_terms", &bewego::MotionObjective::AddWayPointTerms)

      .def("add_sphere", &bewego::MotionObjective::AddSphere)
      .def("add_box", &bewego::MotionObjective::AddBox)
      .def("add_oriented_box", &bewego::MotionObjective::AddOrientedBox)
      .def("clear_workspace", &bewego::MotionObjective::ClearWorkspace)
      .def("set_sdf_gamma", &bewego::MotionObjective::SetSDFGamma)
      .def("set_sdf_margin", &bewego::MotionObjective::SetSDFMargin)

      .def("objective", &bewego::MotionObjective::objective)
      .def("obstacle_potential", &bewego::MotionObjective::obstacle_potential)
      .def("set_verbose", &bewego::MotionObjective::set_verbose);

#ifdef WITH_IPOPT

  namespace opt = bewego::numerical_optimization;

  m.def("test_motion_optimization", &opt::TestMotionOptimization, R"pbdoc(
        return true if can optimize motion with iptop
    )pbdoc");

  py::class_<opt::OptimizeResult>(m, "OptimizeResult")
      .def(py::init<>())
      .def_readonly("x", &opt::OptimizeResult::x)
      .def_readonly("success", &opt::OptimizeResult::success)
      .def_readonly("message", &opt::OptimizeResult::message)
      .def_readonly("status", &opt::OptimizeResult::status)
      .def_readonly("fun", &opt::OptimizeResult::fun)
      .def_readonly("jac", &opt::OptimizeResult::jac)
      .def_readonly("hess", &opt::OptimizeResult::hess)
      .def_readonly("hess_inv", &opt::OptimizeResult::hess_inv)
      .def_readonly("nfev", &opt::OptimizeResult::nfev)
      .def_readonly("njev", &opt::OptimizeResult::njev)
      .def_readonly("nhev", &opt::OptimizeResult::nhev)
      .def_readonly("nit", &opt::OptimizeResult::nit)
      .def_readonly("maxcv", &opt::OptimizeResult::maxcv);

  py::class_<opt::PlanarOptimizer>(m, "PlanarOptimizer")
      .def(py::init<uint32_t, double, const std::vector<double>&>())

      // Main
      .def("optimize", &opt::PlanarOptimizer::Optimize)

      // Constraints
      .def("add_keypoints_surface_constraints",
           &opt::PlanarOptimizer::AddKeyPointsSurfaceConstraints)
      .def("add_smooth_keypoints_surface_constraints",
           &opt::PlanarOptimizer::AddSmoothKeyPointsSurfaceConstraints)
      .def("add_goal_constraint", &opt::PlanarOptimizer::AddGoalConstraint)
      .def("add_goal_manifold_constraint",
           &opt::PlanarOptimizer::AddGoalManifoldConstraint)
      .def("add_waypoint_constraint",
           &opt::PlanarOptimizer::AddWayPointConstraint)
      .def("add_waypoint_manifold_constraint",
           &opt::PlanarOptimizer::AddWayPointManifoldConstraint)

      // Objectives
      .def("add_smoothness_terms", &opt::PlanarOptimizer::AddSmoothnessTerms)
      .def("add_obstacle_terms", &opt::PlanarOptimizer::AddObstacleTerms)
      .def("add_terminal_potential_terms",
           &opt::PlanarOptimizer::AddTerminalPotentialTerms)
      .def("add_waypoint_terms", &opt::PlanarOptimizer::AddWayPointTerms)

      // Workspace
      .def("add_sphere", &opt::PlanarOptimizer::AddSphere)
      .def("add_box", &opt::PlanarOptimizer::AddBox)
      .def("add_oriented_box", &bewego::MotionObjective::AddOrientedBox)
      .def("clear_workspace", &opt::PlanarOptimizer::ClearWorkspace)
      .def("set_sdf_gamma", &opt::PlanarOptimizer::SetSDFGamma)
      .def("set_sdf_margin", &opt::PlanarOptimizer::SetSDFMargin)

      // Functions
      .def("set_trajectory_publisher",
           &opt::PlanarOptimizer::set_trajectory_publisher)
      .def("objective", &opt::PlanarOptimizer::objective)
      .def("obstacle_potential", &opt::PlanarOptimizer::obstacle_potential)
      .def("set_verbose", &opt::PlanarOptimizer::set_verbose);

  py::class_<opt::FreeflyerOptimzer>(m, "FreeflyerOptimzer")
      .def(py::init<uint32_t, uint32_t, double, const std::vector<double>&,
                    std::string, const std::vector<Eigen::VectorXd>&,
                    const std::vector<double>&>())

      // Main
      .def("optimize", &opt::FreeflyerOptimzer::Optimize)

      // Constraints
      .def("add_keypoints_surface_constraints",
           &opt::FreeflyerOptimzer::AddKeyPointsSurfaceConstraints)
      .def("add_smooth_keypoints_surface_constraints",
           &opt::FreeflyerOptimzer::AddKeyPointsSurfaceConstraints)
      .def("add_goal_constraint", &opt::FreeflyerOptimzer::AddGoalConstraint)
      .def("add_waypoint_constraint",
           &opt::FreeflyerOptimzer::AddWayPointConstraint)

      // Objectives
      .def("add_smoothness_terms", &opt::FreeflyerOptimzer::AddSmoothnessTerms)
      .def("add_obstacle_terms", &opt::FreeflyerOptimzer::AddObstacleTerms)
      .def("add_terminal_potential_terms",
           &opt::FreeflyerOptimzer::AddTerminalPotentialTerms)
      .def("add_waypoint_terms", &opt::FreeflyerOptimzer::AddWayPointTerms)

      // Workspace
      .def("add_sphere", &opt::FreeflyerOptimzer::AddSphere)
      .def("add_box", &opt::FreeflyerOptimzer::AddBox)
      .def("add_oriented_box", &bewego::MotionObjective::AddOrientedBox)
      .def("clear_workspace", &opt::FreeflyerOptimzer::ClearWorkspace)
      .def("set_sdf_gamma", &opt::FreeflyerOptimzer::SetSDFGamma)
      .def("set_sdf_margin", &opt::FreeflyerOptimzer::SetSDFMargin)

      // Functions
      .def("set_trajectory_publisher",
           &opt::FreeflyerOptimzer::set_trajectory_publisher)
      .def("objective", &opt::FreeflyerOptimzer::objective)
      .def("obstacle_potential", &opt::FreeflyerOptimzer::obstacle_potential)

      // Options
      .def("set_q_default", &opt::FreeflyerOptimzer::set_q_default)
      .def("set_end_effector", &opt::FreeflyerOptimzer::set_end_effector)
      .def("set_clique_collision_constraints",
           &opt::FreeflyerOptimzer::set_clique_collision_constraints)
      .def("set_verbose", &opt::FreeflyerOptimzer::set_verbose);

  py::class_<opt::RobotOptimizer>(m, "RobotOptimizer")
      .def(py::init<uint32_t, uint32_t, double, const std::vector<double>&,
                    std::shared_ptr<const bewego::Robot>>())

      // Main
      .def("optimize", &opt::RobotOptimizer::Optimize)

      // Constraints
      .def("add_keypoints_surface_constraints",
           &opt::RobotOptimizer::AddKeyPointsSurfaceConstraints)
      .def("add_smooth_keypoints_surface_constraints",
           &opt::RobotOptimizer::AddKeyPointsSurfaceConstraints)
      .def("add_goal_constraint", &opt::RobotOptimizer::AddGoalConstraint)
      .def("add_waypoint_constraint",
           &opt::RobotOptimizer::AddWayPointConstraint)

      // Objectives
      .def("add_smoothness_terms", &opt::RobotOptimizer::AddSmoothnessTerms)
      .def("add_obstacle_terms", &opt::RobotOptimizer::AddGeodesicTerms)
      .def("add_terminal_potential_terms",
           &opt::RobotOptimizer::AddTerminalPotentialTerms)
      .def("add_terminal_endeffector_potential_terms",
           &opt::RobotOptimizer::AddTerminalEndeffectorPotentialTerms)
      .def("add_waypoint_terms", &opt::RobotOptimizer::AddWayPointTerms)

      // Workspace
      .def("add_sphere", &opt::RobotOptimizer::AddSphere)
      .def("add_box", &opt::RobotOptimizer::AddBox)
      .def("add_oriented_box", &bewego::MotionObjective::AddOrientedBox)
      .def("clear_workspace", &opt::RobotOptimizer::ClearWorkspace)
      .def("set_sdf_gamma", &opt::RobotOptimizer::SetSDFGamma)
      .def("set_sdf_margin", &opt::RobotOptimizer::SetSDFMargin)

      // Functions
      .def("set_trajectory_publisher",
           &opt::RobotOptimizer::set_trajectory_publisher)
      .def("objective", &opt::RobotOptimizer::objective)
      .def("obstacle_potential", &opt::RobotOptimizer::obstacle_potential)

      // Options
      .def("set_q_default", &opt::RobotOptimizer::set_q_default)
      .def("set_end_effector", &opt::RobotOptimizer::set_end_effector)
      .def("set_clique_collision_constraints",
           &opt::RobotOptimizer::set_clique_collision_constraints)
      .def("set_verbose", &opt::RobotOptimizer::set_verbose);
#endif

  m.attr("__version__") = "0.0.1";
}