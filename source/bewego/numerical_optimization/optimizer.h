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
 *                                              Jim Mainprice Tue 18 Nov 2019
 */
#pragma once

#include <bewego/numerical_optimization/constrained_optimization_problem.h>

#include <Eigen/Core>

namespace bewego {
namespace numerical_optimization {

/*
     Represents the optimization result.
     This is a copy of the structure used by scipy

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    */
struct OptimizeResult {
  OptimizeResult()
      : success(false),
        status(-1),
        nfev(0),
        njev(0),
        nhev(0),
        nit(0),
        maxcv(std::numeric_limits<double>::max()) {}

  Eigen::VectorXd x;    // The solution of the optimization.
  bool success;         // Whether or not the optimizer exited successfully.
  std::string message;  // Description of the cause of the termination.
  int status;  // Termination status of the optimizer. Its value depends on the
               // underlying solver. Refer to `message` for details.

  // fun, jac, hess
  // Values of objective function, its Jacobian and its Hessian (if
  //       available). The Hessians may be approximations, see the documentation
  //       of the function in question.
  Eigen::VectorXd fun;
  Eigen::MatrixXd jac;
  Eigen::MatrixXd hess;

  // Inverse of the objective function's Hessian; may be an approximation.
  // Not available for all solvers. The type of this attribute may be
  // either np.ndarray or scipy.sparse.linalg.LinearOperator.
  Eigen::MatrixXd hess_inv;

  // Number of evaluations of the objective functions and of its
  //       Jacobian and Hessian.
  uint32_t nfev, njev, nhev;

  uint32_t nit;  // Number of iterations performed by the optimizer.
  double maxcv;  // The maximum constraint violation.
};

class LocalSolution {
 public:
  enum WarningCode { NO_WARNING = 0, DID_NOT_CONVERGE };
  LocalSolution() : success_(false), error_str_("Uninitialized") {}
  LocalSolution(const Eigen::VectorXd& x, double objective_value,
                const Eigen::VectorXd& gradient, const Eigen::MatrixXd& hessian,
                WarningCode warning_code = NO_WARNING)
      : success_(true),
        warning_code_(warning_code),
        x_(x),
        objective_value_(objective_value),
        gradient_(gradient),
        hessian_(hessian) {}

  LocalSolution(const LocalSolution& solution)
      : success_(solution.success_),
        error_str_(solution.error_str_),
        warning_code_(solution.warning_code_),
        x_(solution.x_),
        objective_value_(solution.objective_value_),
        gradient_(solution.gradient_),
        hessian_(solution.hessian_) {}

  bool success() const { return success_; }
  const std::string& error_str() const { return error_str_; }
  WarningCode warning_code() const { return warning_code_; }
  const Eigen::VectorXd x() const { return x_; }
  double objective_value() const { return objective_value_; }
  Eigen::VectorXd gradient() const { return gradient_; }
  Eigen::MatrixXd hessian() const { return hessian_; }
  bool HasWarning() const { return warning_code_ != NO_WARNING; }

 protected:
  LocalSolution(const std::string& error_str)
      : success_(false), error_str_(error_str) {}

 private:
  bool success_;
  std::string error_str_;
  WarningCode warning_code_;
  Eigen::VectorXd x_;
  double objective_value_;
  Eigen::VectorXd gradient_;
  Eigen::MatrixXd hessian_;
};

class LocalOptimizer {
 public:
  LocalOptimizer() : verbose_(false) {}
  virtual LocalSolution Run(const DifferentiableMap& f,
                            const Eigen::VectorXd& x0) const = 0;

  void set_verbose(bool v) { verbose_ = v; }

  /*!\brief This function sets a value atouside the optimizer
   * for visualiztion and logging purpuses.
   * TODO test overhead and take out of verbose mode.
   * */
  void set_current_solution_accessor(
      std::function<void(const Eigen::VectorXd&)> f) {
    publish_solution_ = f;
  }

 protected:
  bool verbose_;
  std::function<void(const Eigen::VectorXd&)> publish_solution_;
};

class ConstrainedSolution {
 public:
  enum WarningCode { NO_WARNING = 0, DID_NOT_CONVERGE };
  ConstrainedSolution() : success_(false), error_str_("Uninitialized") {}
  ConstrainedSolution(const Eigen::VectorXd& x, double objective_value,
                      const Eigen::VectorXd& inequality_constraint_values,
                      const Eigen::VectorXd& equality_constraint_values,
                      bool success,
                      WarningCode warning_code = NO_WARNING)
      : success_(success),
        warning_code_(warning_code),
        x_(x),
        objective_value_(objective_value),
        inequality_constraint_values_(inequality_constraint_values),
        equality_constraint_values_(equality_constraint_values) {}

  ConstrainedSolution(const ConstrainedSolution& solution)
      : success_(solution.success_),
        error_str_(solution.error_str_),
        warning_code_(solution.warning_code_),
        x_(solution.x_),
        objective_value_(solution.objective_value_),
        inequality_constraint_values_(solution.inequality_constraint_values_),
        equality_constraint_values_(solution.equality_constraint_values_) {}

  bool success() const { return success_; }
  const std::string& error_str() const { return error_str_; }
  WarningCode warning_code() const { return warning_code_; }

  const Eigen::VectorXd x() const { return x_; }
  double objective_value() const { return objective_value_; }

  const Eigen::VectorXd& inequality_constraint_values() const {
    return inequality_constraint_values_;
  }
  const Eigen::VectorXd& equality_constraint_values() const {
    return equality_constraint_values_;
  }

  bool HasInequalityConstraints() const {
    return inequality_constraint_values_.rows() > 0;
  }
  bool HasEqualityConstraints() const {
    return equality_constraint_values_.rows() > 0;
  }
  bool HasWarning() const { return warning_code_ != NO_WARNING; }

 protected:
  ConstrainedSolution(const std::string& error_str)
      : success_(false), error_str_(error_str) {}

 private:
  bool success_;
  std::string error_str_;
  WarningCode warning_code_;
  Eigen::VectorXd x_;
  double objective_value_;
  Eigen::VectorXd inequality_constraint_values_;
  Eigen::VectorXd equality_constraint_values_;
  friend class Error;
};

class Error : public ConstrainedSolution {
 public:
  Error(const std::string& error_str) : ConstrainedSolution(error_str) {}
};

class ConstrainedOptimizer {
 public:
  //

  ConstrainedOptimizer() : verbose_(false) {}

  void set_verbose(bool value) { verbose_ = value; }
  bool verbose() { return verbose_; }

  virtual ConstrainedSolution Run(const ConstrainedOptimizationProblem& problem,
                                  const Eigen::VectorXd& x0) const = 0;

 protected:
  bool verbose_;
};

}  // namespace numerical_optimization
}  // namespace bewego