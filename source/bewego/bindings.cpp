// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/planar_grid.h>
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

  // Setters.
  void set_pace(double v) {
    problem_->set_pace(v);
  }
  void set_env_size(std::vector<double> v) {
    if (v.size() != 4) {
      throw std::runtime_error("Input shapes must be 4");
    }
    cout << "set env size : " 
        << v[0] << " , " << v[1] << " , " 
        << v[2] << " , " << v[3] << endl;
     problem_->set_env_size(v);
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
      .def("set_pace", &bewego::AStarGrid::set_pace)
      .def("set_env_size", &bewego::AStarGrid::set_env_size)
      .def("pace", &bewego::AStarGrid::pace)
      .def("env_size", &bewego::AStarGrid::env_size);

  m.attr("__version__") = "0.0.1";
}