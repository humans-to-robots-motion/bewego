// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/planar_grid.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
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
  AStarGrid() {
    problem_ = std::make_shared<AStarProblem>();
  }

  // Getters.
  double pace() const {
    return problem_->pace();
  }
  std::vector<double> env_size() const {
     return problem_->env_size();
  }

  void init_grid(double p, const std::vector<double>& e){
    problem_->set_pace(p);
    problem_->set_env_size(e);
    problem_->InitGrid();
  }
  void set_costs(const Eigen::MatrixXd& costs) {
     problem_->InitCosts(costs);
  }

protected:
    std::shared_ptr<AStarProblem> problem_;
};

}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


PYBIND11_MODULE(pybewego, m) {
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

  py::class_<bewego::AStarGrid>(m, "AStarGrid")
      .def(py::init<>())
      .def("pace", &bewego::AStarGrid::pace)
      .def("env_size", &bewego::AStarGrid::env_size)
      .def("init_grid", &bewego::AStarGrid::init_grid)
      .def("set_costs", &bewego::AStarGrid::set_costs);

  m.attr("__version__") = "0.0.1";
}