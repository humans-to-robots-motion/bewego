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

#pragma once

#include <bewego/numerical_optimization/constrained_optimization_problem.h>
#include <bewego/numerical_optimization/ipopt_optimizer.h>
#include <bewego/util/bounds.h>
#include <bewego/util/misc.h>

#include <Eigen/Core>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

/**
 * @brief Solves the optimization problem using the IPOPT solver.
 *
 * Given an optimization Problem with variables, costs and constraints, this
 * class wraps it and makes it conform with the interface defined by IPOPT
 * https://projects.coin-or.org/Ipopt
 *
 * This implements the Adapter pattern. This class should not add any
 * functionality, but merely delegate it to the Adaptee (the Problem object).
 */
namespace bewego {
namespace numerical_optimization {

class IpoptProblem : public Ipopt::TNLP {
 public:
  using VectorXd = Eigen::VectorXd;
  using Jacobian = Eigen::MatrixXd;
  using NonLinearProblemPtr =
      std::shared_ptr<const OptimizationProblemWithConstraints>;

  /**
   * @brief  Creates an IpoptAdapter wrapping the @a nlp.
   * @param  nlp  The specific nonlinear programming problem.
   *
   * This constructor holds and modifies the passed nlp.
   */
  IpoptProblem(NonLinearProblemPtr nlp,
               const std::vector<util::Bounds>& bounds_x,
               const std::vector<util::Bounds>& bounds_g,
               const Eigen::VectorXd& x0, bool finite_diff = false,
               std::function<void(const Eigen::VectorXd&)> function =
                   std::function<void(const Eigen::VectorXd&)>());
  virtual ~IpoptProblem() = default;

  // Get solution
  Eigen::VectorXd solution() const { return x_solution_; }
  void set_verbose(bool v) { verbose_ = v; }
  void set_hessian_sparcity_patern(const util::MatrixSparsityPatern& v) {
    hessian_sparcity_patern_ = v;
  }
  void set_g_sparcity_patern(const std::vector<util::MatrixSparsityPatern>& v) {
    g_gradient_sparcity_paterns_ = v;
  }
  void set_h_sparcity_patern(const std::vector<util::MatrixSparsityPatern>& v) {
    h_gradient_sparcity_paterns_ = v;
  }

 protected:
  // output
  bool verbose_;
  std::function<void(const Eigen::VectorXd&)> publish_solution_;

  // problem
  uint32_t input_dimension_;
  uint32_t n_g_;
  uint32_t n_h_;
  NonLinearProblemPtr nlp_;
  std::vector<util::Bounds> bounds_x_;
  std::vector<util::Bounds> bounds_g_;
  bool finite_diff_;  ///< Flag that indicates the "finite-difference-values"

  std::vector<util::MatrixSparsityPatern> g_gradient_sparcity_paterns_;
  std::vector<util::MatrixSparsityPatern> h_gradient_sparcity_paterns_;
  util::MatrixSparsityPatern hessian_sparcity_patern_;

  /// solution
  VectorXd x0_;
  std::vector<VectorXd> x_prev_;
  VectorXd x_solution_;

  // Return an Eigen vector of the paramters
  Eigen::VectorXd get_x(const double* x) const {
    return Eigen::Map<const Eigen::VectorXd>(
        x, nlp_->objective_function()->input_dimension());
  }

  /** Test nlp finite differences **/
  void test_constraints_finite_differences(const Eigen::VectorXd& x) const;

  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                            Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                            IndexStyleEnum& index_style);

  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Ipopt::Index n, double* x_l, double* x_u,
                               Ipopt::Index m, double* g_l, double* g_u);

  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Ipopt::Index n, bool init_x, double* x,
                                  bool init_z, double* z_L, double* z_U,
                                  Ipopt::Index m, bool init_lambda,
                                  double* lambda);

  /** Method to return the objective value */
  virtual bool eval_f(Ipopt::Index n, const double* x, bool new_x,
                      double& obj_value);

  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Ipopt::Index n, const double* x, bool new_x,
                           double* grad_f);

  /** Method to return the constraint residuals */
  virtual bool eval_g(Ipopt::Index n, const double* x, bool new_x,
                      Ipopt::Index m, double* g);

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Ipopt::Index n, const double* x, bool new_x,
                          Ipopt::Index m, Ipopt::Index nele_jac,
                          Ipopt::Index* iRow, Ipopt::Index* jCol,
                          double* values);

  /** This is called after every iteration and is used to save intermediate
   *  solutions in the nlp */
  virtual bool intermediate_callback(Ipopt::AlgorithmMode mode,
                                     Ipopt::Index iter, double obj_value,
                                     double inf_pr, double inf_du, double mu,
                                     double d_norm, double regularization_size,
                                     double alpha_du, double alpha_pr,
                                     Ipopt::Index ls_trials,
                                     const Ipopt::IpoptData* ip_data,
                                     Ipopt::IpoptCalculatedQuantities* ip_cq);

  /** This method is called when the algorithm is complete so the TNLP can
   * store/write the solution */
  virtual void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n,
                                 const double* x, const double* z_L,
                                 const double* z_U, Ipopt::Index m,
                                 const double* g, const double* lambda,
                                 double obj_value,
                                 const Ipopt::IpoptData* ip_data,
                                 Ipopt::IpoptCalculatedQuantities* ip_cq);

  /** overload this method to return the hessian of the
   *  lagrangian. The vectors iRow and jCol only need to be set once
   *  (during the first call). The first call is used to set the
   *  structure only (iRow and jCol will be non-NULL, and values
   *  will be NULL) For subsequent calls, iRow and jCol will be
   *  NULL. This matrix is symmetric - specify the lower diagonal
   *  only.  A default implementation is provided, in case the user
   *  wants to se quasi-Newton approximations to estimate the second
   *  derivatives and doesn't not neet to implement this method. */
  virtual bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                      Ipopt::Number obj_factor, Ipopt::Index m,
                      const Ipopt::Number* lambda, bool new_lambda,
                      Ipopt::Index nele_hess, Ipopt::Index* iRow,
                      Ipopt::Index* jCol, Ipopt::Number* values);
};

}  // namespace numerical_optimization
}  // namespace bewego
