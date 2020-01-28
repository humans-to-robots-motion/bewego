// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/rnn.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using namespace bewego;

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

  m.def("test_identity", &test_identity, R"pbdoc(
        Test the identity map
        Some other explanation about the subtract function.
    )pbdoc");
  py::class_<RNNCell, std::shared_ptr<RNNCell>>(m, "RNNCell");
  py::class_<CoupledRNNCell, std::shared_ptr<CoupledRNNCell>>(m, "CoupledRNNCell");

  py::class_<GRUCell, CoupledRNNCell, std::shared_ptr<GRUCell>>(m, "GRUCell")
    .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&>())
    .def("Forward", &GRUCell::Forward)
    .def("Jacobian", &GRUCell::Jacobian)
    .def("output_dimension", &GRUCell::output_dimension)
    .def("hidden_dimension", &GRUCell::hidden_dimension)
    .def("input_dimension", &GRUCell::input_dimension)
    ;
  py::class_<StackedCoupledRNNCell, RNNCell, std::shared_ptr<StackedCoupledRNNCell>>(m, "StackedCoupledRNNCell")
    .def(py::init<const int, const int, const int, const std::vector<std::shared_ptr<CoupledRNNCell>>& , const Eigen::MatrixXd&, const Eigen::VectorXd&>())
    .def("Forward", &StackedCoupledRNNCell::Forward)
    .def("Jacobian", &StackedCoupledRNNCell::Jacobian)
    .def("output_dimension", &StackedCoupledRNNCell::output_dimension)
    .def("hidden_dimension", &StackedCoupledRNNCell::hidden_dimension)
    .def("input_dimension", &StackedCoupledRNNCell::input_dimension)
    ;
  py::class_<VRED>(m, "VRED")
    .def(py::init<std::shared_ptr<RNNCell>&, int>())
    .def("Forward", &VRED::Forward)
    .def("Jacobian", &VRED::Jacobian)
    ;
  m.attr("__version__") = "0.0.1";
}
