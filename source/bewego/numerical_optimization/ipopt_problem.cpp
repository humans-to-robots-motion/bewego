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
 *                                              Jim Mainprice We 18 Dec 2019
 */

#include <bewego/numerical_optimization/ipopt_problem.h>

using namespace Ipopt;
using namespace bewego;
using namespace bewego::numerical_optimization;
using std::cout;
using std::endl;

IpoptProblem::IpoptProblem(NonLinearProblemPtr nlp,
                           const std::vector<Bounds>& bounds_x,
                           const std::vector<Bounds>& bounds_g,
                           const Eigen::VectorXd& x0, bool finite_diff,
                           std::function<void(const Eigen::VectorXd&)> function)
    : verbose_(true),
      publish_solution_(function),
      input_dimension_(nlp->objective_function()->input_dimension()),
      n_g_(nlp->num_inequality_constraints()),
      n_h_(nlp->num_equality_constraints()),
      nlp_(nlp),
      bounds_x_(bounds_x),
      bounds_g_(bounds_g),
      finite_diff_(finite_diff),
      x0_(x0) {}

bool IpoptProblem::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                Index& nnz_h_lag, IndexStyleEnum& index_style) {
  n = input_dimension_;
  m = n_g_ + n_h_;
  /* For sparse nnz_jac_g = nlp_->GetJacobianOfConstraints().nonZeros();
   */
  nnz_jac_g = m * n;  // dense jacobian
  if (hessian_sparcity_patern_.empty()) {
    nnz_h_lag = n * (n + 1) / 2;  // lower triange hessian (can not change that)
  } else {
    uint32_t nnz_diag = hessian_sparcity_patern_.nb_diag_terms();
    uint32_t nnz_offd = hessian_sparcity_patern_.nb_offdiag_terms();
    nnz_h_lag = nnz_diag + nnz_offd / 2;
  }

  // start index at 0 for row/col entries
  index_style = C_STYLE;
  return true;
}

bool IpoptProblem::get_bounds_info(Index n, double* x_lower, double* x_upper,
                                   Index m, double* g_l, double* g_u) {
  for (uint c = 0; c < bounds_x_.size(); ++c) {
    x_lower[c] = bounds_x_[c].lower_;
    x_upper[c] = bounds_x_[c].upper_;
  }
  // specific bounds depending on equality and inequality constraints
  for (uint c = 0; c < bounds_g_.size(); ++c) {
    g_l[c] = bounds_g_[c].lower_;
    g_u[c] = bounds_g_[c].upper_;
  }
  return true;
}

bool IpoptProblem::get_starting_point(Index n, bool init_x, double* x,
                                      bool init_z, double* z_L, double* z_U,
                                      Index m, bool init_lambda,
                                      double* lambda) {
  // Here, we assume we only have starting values for x
  assert(init_x == true);
  assert(init_z == false);
  assert(init_lambda == false);
  Eigen::Map<VectorXd>(&x[0], x0_.rows()) = x0_;
  return true;
}

bool IpoptProblem::eval_f(Index n, const double* x, bool new_x,
                          double& obj_value) {
  Eigen::VectorXd x_t = get_x(x);
  obj_value = nlp_->objective_function()->ForwardFunc(x_t);
  if (verbose_ && publish_solution_) {
    // cout << "publish solution... (" << x_t.size() << ")" << endl;
    // cout << "sol : " << x_t.transpose() << endl;
    publish_solution_(x_t);
  }
  return true;
}

bool IpoptProblem::eval_grad_f(Index n, const double* x, bool new_x,
                               double* grad_f) {
  Eigen::VectorXd x_t = get_x(x);
  Eigen::VectorXd grad = nlp_->objective_function()->Gradient(x_t);
  Eigen::Map<Eigen::MatrixXd>(grad_f, n, 1) = grad;
  // Set the current solution to external thread for visualization
  // or statistics
  return true;
}

bool IpoptProblem::eval_g(Index n, const double* x, bool new_x, Index m,
                          double* g) {
  assert(m == n_g_ + n_h_);
  assert(n == input_dimension_);
  VectorXd g_eig = Eigen::VectorXd(m);
  Eigen::VectorXd x_t = get_x(x);
  for (uint32_t i = 0; i < n_g_; i++) {
    g_eig[i] = nlp_->inequality_constraints()[i]->ForwardFunc(x_t);
  }
  for (uint32_t i = n_g_; i < n_h_ + n_g_; i++) {
    g_eig[i] = nlp_->equality_constraints()[i - n_g_]->ForwardFunc(x_t);
  }
  Eigen::Map<VectorXd>(g, m) = g_eig;
  return true;
}

bool IpoptProblem::eval_jac_g(Index n, const double* x, bool new_x, Index m,
                              Index nele_jac, Index* iRow, Index* jCol,
                              double* values) {
  assert(m == n_g_ + n_h_);
  assert(n == input_dimension_);
  // defines the positions of the nonzero elements of the jacobian
  if (values == nullptr) {
    // initial sparsity structure is never allowed to change
    // If "jacobian_approximation" option is set as "finite-difference-values",
    // the Jacobian is dense!
    Index nele = 0;
    // Dense jacobian, defined as ColMajor patern (default Eigen patern)
    for (Index col = 0; col < n; col++) {
      for (Index row = 0; row < m; row++) {
        iRow[nele] = row;
        jCol[nele] = col;
        nele++;
      }
    }
    assert(nele == nele_jac);
  } else {
    // only gets used if "jacobian_approximation finite-difference-values" is
    // not set defines the positions of the nonzero elements of the jacobian
    // If "jacobian_approximation" option is set as "finite-difference-values",
    // the Jacobian is dense!
    // By default Eigen matrices are ColMajor
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Eigen::VectorXd x_t = get_x(x);
    Eigen::VectorXd grad(n);
    Eigen::VectorXd grad_diff(n);
    Eigen::MatrixXd jacobian(m, n);

    // test_finite_differences(x_t);
    for (uint32_t i = 0; i < n_g_; i++) {
      const auto& g = *nlp_->inequality_constraints()[i];
      jacobian.row(i) = g.Gradient(x_t);
    }
    for (uint32_t i = n_g_; i < n_h_ + n_g_; i++) {
      const auto& h = *nlp_->equality_constraints()[i - n_g_];
      jacobian.row(i) = h.Gradient(x_t);
    }

    Eigen::Map<Eigen::MatrixXd>(values, m, n) = jacobian;
  }
  return true;
}

void IpoptProblem::test_constraints_finite_differences(
    const Eigen::VectorXd& x) const {
  Eigen::VectorXd grad(input_dimension_);
  Eigen::VectorXd grad_diff(input_dimension_);

  // Objective
  const auto& f = *nlp_->objective_function();
  grad = f.Gradient(x);
  grad_diff = DifferentiableMap::FiniteDifferenceJacobian(f, x).row(0);
  double delta = (grad - grad_diff).norm();
  assert(delta < 1e-6);

  // Inequalities
  for (uint32_t i = 0; i < n_g_; i++) {
    const auto& g = *nlp_->inequality_constraints()[i];
    grad = g.Gradient(x);
    grad_diff = DifferentiableMap::FiniteDifferenceJacobian(g, x).row(0);
    delta = (grad - grad_diff).norm();
    assert(delta < 1e-6);
  }

  // Equalities
  for (uint32_t i = 0; i < n_h_; i++) {
    const auto& h = *nlp_->equality_constraints()[i];
    grad = h.Gradient(x);
    grad_diff = DifferentiableMap::FiniteDifferenceJacobian(h, x).row(0);
    delta = (grad - grad_diff).norm();
    assert(delta < 1e-6);
  }
}

/** overload this method to return the hessian of the
 *  lagrangian. The vectors iRow and jCol only need to be set once
 *  (during the first call). The first call is used to set the
 *  structure only (iRow and jCol will be non-NULL, and values
 *  will be NULL) For subsequent calls, iRow and jCol will be
 *  NULL. This matrix is symmetric - specify the lower diagonal
 *  only.  A default implementation is provided, in case the user
 *  wants to se quasi-Newton approximations to estimate the second
 *  derivatives and doesn't not neet to implement this method. */
bool IpoptProblem::eval_h(Index n, const Number* x, bool new_x,
                          Number obj_factor, Index m, const Number* lambda,
                          bool new_lambda, Index nele_hess, Index* iRow,
                          Index* jCol, Number* values) {
  assert(n == input_dimension_);
  assert(m == n_g_ + n_h_);

  if (values == nullptr) {
    // initial sparsity structure is never allowed to change
    if (!hessian_sparcity_patern_.empty()) {
      Index nele = 0;
      // Define a ColMajor patern
      for (Index col = 0; col < n; col++) {
        for (Index row = 0; row <= col; row++) {
          iRow[nele] = row;
          jCol[nele] = col;
          nele++;
        }
      }
    } else {
      // initial sparsity structure is never allowed to change
      Index nele = 0;
      // Define a ColMajor patern
      for (Index col = 0; col < n; col++) {
        for (Index row = 0; row <= col; row++) {
          iRow[nele] = row;
          jCol[nele] = col;
          nele++;
        }
      }
    }
    assert(nele == nele_hess);
  } else {
    // only gets used if "jacobian_approximation finite-difference-values" is
    // not set defines the positions of the nonzero elements of the jacobian
    // If "jacobian_approximation" option is set as "finite-difference-values",
    // the Jacobian is dense!
    // By default Eigen matrices are ColMajor
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Eigen::VectorXd x_t = get_x(x);
    Eigen::VectorXd grad(n);
    Eigen::MatrixXd H(n, n);
    grad = nlp_->objective_function()->Gradient(x_t);
    H = nlp_->objective_function()->Hessian(x_t);

    H *= obj_factor;  // TODO: Don't know why..

    Eigen::MatrixXd Hess(n, n);
    // Inequalities
    for (uint32_t i = 0; i < n_g_; i++) {
      const auto& g = *nlp_->inequality_constraints()[i];
      Hess = g.Hessian(x_t);
      H += lambda[i] * Hess;
    }
    // Equalities
    for (uint32_t i = n_g_; i < n_h_ + n_g_; i++) {
      const auto& h = *nlp_->equality_constraints()[i - n_g_];
      Hess = h.Hessian(x_t);
      H += lambda[i] * Hess;
    }

    // Copy lower triangle of augmented-Lagrangian Hessian matrix
    Index nele = 0;
    for (Index col = 0; col < n; col++) {
      for (Index row = 0; row <= col; row++) {
        values[nele++] = H(row, col);
      }
    }
    assert(nele == nele_hess);
  }
  return true;
}

bool IpoptProblem::intermediate_callback(
    AlgorithmMode mode, Index iter, double obj_value, double inf_pr,
    double inf_du, double mu, double d_norm, double regularization_size,
    double alpha_du, double alpha_pr, Index ls_trials, const IpoptData* ip_data,
    IpoptCalculatedQuantities* ip_cq) {
  x_prev_.push_back(x_solution_);
  return true;
}

void IpoptProblem::finalize_solution(SolverReturn status, Index n,
                                     const double* x, const double* z_L,
                                     const double* z_U, Index m,
                                     const double* g, const double* lambda,
                                     double obj_value, const IpoptData* ip_data,
                                     IpoptCalculatedQuantities* ip_cq) {
  x_solution_ = get_x(x);
  x_prev_.push_back(x_solution_);
}
