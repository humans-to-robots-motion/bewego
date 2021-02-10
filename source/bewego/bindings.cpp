// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/planar_grid.h>
#include <bewego/value_iteration.h>
#include <bewego/geometry.h>
#include <bewego/kinematics.h>
#include <bewego/util.h>

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

  m.attr("__version__") = "0.0.1";
}