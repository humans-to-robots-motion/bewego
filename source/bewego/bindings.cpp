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
#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/geometry.h>
#include <bewego/kinematics.h>
#include <bewego/objective.h>
#include <bewego/planar_grid.h>
#include <bewego/trajectory.h>
#include <bewego/util.h>
#include <bewego/value_iteration.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
using std::cout;
using std::endl;

int add(int i, int j) { return i + j; }

bool test_identity(int n) {
  uint32_t dimension = n;
  auto f = std::make_shared<bewego::IdentityMap>(dimension);
  auto x = Eigen::VectorXd::Random(dimension);
  auto J1 = f->Jacobian(x);
  auto J2 = bewego::DifferentiableMap::FiniteDifferenceJacobian(*f, x);
  bool a = J1.isApprox(J2);
  bool b = true;
  if (n == 1) {
    auto H1 = f->Hessian(x);
    auto H2 = bewego::DifferentiableMap::FiniteDifferenceHessian(*f, x);
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

  m.def("quaternion_to_matrix", &QuaternionToMatrix, R"pbdoc(
        Returns the rotation matrix corresponding to a Quaternion
    )pbdoc");

  m.def("euler_to_quaternion", &EulerToQuaternion, R"pbdoc(
        Returns the quaternion corresponding to euler
    )pbdoc");

  py::class_<bewego::Robot>(m, "Robot")
      .def(py::init<>())
      .def("add_rigid_body", &bewego::Robot::AddRigidBody)
      .def("set_and_update", &bewego::Robot::SetAndUpdate)
      .def("get_position", &bewego::Robot::get_position)
      .def("get_rotation", &bewego::Robot::get_rotation)
      .def("get_transform", &bewego::Robot::get_transform)
      .def("get_jacobian", &bewego::Robot::JacobianPosition)
      .def("get_jacobian_axis", &bewego::Robot::JacobianAxis)
      .def("get_jacobian_frame", &bewego::Robot::JacobianFrame)
      .def("set_base_transform", &bewego::Robot::set_base_transform);

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
      .def("hessian", &bewego::TrajectoryObjectiveFunction::Hessian);

  py::class_<bewego::MotionObjective>(m, "MotionObjective")
      .def(py::init<uint32_t, double, uint32_t>())
      .def("add_smoothness_terms", &bewego::MotionObjective::AddSmoothnessTerms)
      .def("add_obstacle_terms", &bewego::MotionObjective::AddObstacleTerms)
      .def("add_terminal_potential_terms",
           &bewego::MotionObjective::AddTerminalPotentialTerms)
      .def("add_waypoint_terms", &bewego::MotionObjective::AddWayPointTerms)
      .def("add_sphere", &bewego::MotionObjective::AddSphere)
      .def("add_box", &bewego::MotionObjective::AddBox)
      .def("clear_workspace", &bewego::MotionObjective::ClearWorkspace)
      .def("objective", &bewego::MotionObjective::objective);

  m.attr("__version__") = "0.0.1";
}