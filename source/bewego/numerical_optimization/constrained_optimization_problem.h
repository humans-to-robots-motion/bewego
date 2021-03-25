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

#include <bewego/derivatives/differentiable_map.h>
#include <bewego/util/misc.h>

namespace bewego {
namespace numerical_optimization {

//! variable bounds
struct Bounds {
  Bounds(double lower = 0.0, double upper = 0.0)
      : lower_(lower), upper_(upper) {}
  double lower_;
  double upper_;
};

//!\brief Clips vector x to bounds v
void BoundClip(const std::vector<Bounds>& v, Eigen::VectorXd* x);

//!\brief Constraint Optimization Problem class.
class ConstrainedOptimizationProblem {
 public:
  /*!\brief Evaluates both the objective and constraints
   *
   * Returns the objective value as the return parameter and the inequality and
   * inequality constraint function evaluations as g_evaluations and
   * h_evaluations, respectively.
   */
  virtual double Evaluate(const Eigen::VectorXd& x,
                          Eigen::VectorXd* g_evaluations,
                          Eigen::VectorXd* h_evaluations) const = 0;

  double operator()(const Eigen::VectorXd& x, Eigen::VectorXd* g_evaluations,
                    Eigen::VectorXd* h_evaluations) const {
    return Evaluate(x, g_evaluations, h_evaluations);
  }

  virtual uint32_t num_inequality_constraints() const = 0;
  virtual uint32_t num_equality_constraints() const = 0;
  uint32_t num_constraints() const {
    return num_inequality_constraints() + num_equality_constraints();
  }
};

/*!\brief This version of the constrained optimization problem implements
 * a simple ConstrainedOptimizationProblem using generic differentiable
 * functions for the inequality and equality constraints.
 */
class OptimizationProblemWithConstraints
    : public ConstrainedOptimizationProblem {
 public:
  OptimizationProblemWithConstraints() {}
  OptimizationProblemWithConstraints(
      DifferentiableMapPtr objective_terms,
      const std::vector<DifferentiableMapPtr>& inequality_constraints,
      const std::vector<DifferentiableMapPtr>& equality_constraints);

  virtual ~OptimizationProblemWithConstraints();

  double Evaluate(const Eigen::VectorXd& x,
                  Eigen::VectorXd* inequality_evaluations,
                  Eigen::VectorXd* equality_evaluations) const override;

  virtual uint32_t num_inequality_constraints() const override { return n_g_; }
  virtual uint32_t num_equality_constraints() const override { return n_h_; }

  DifferentiableMapPtr objective_function() const {
    return objective_function_;
  }
  const std::vector<DifferentiableMapPtr>& inequality_constraints() const {
    return inequality_constraints_;
  }
  const std::vector<DifferentiableMapPtr>& equality_constraints() const {
    return equality_constraints_;
  }

  void add_inequality_constraint(DifferentiableMapPtr g);
  void add_equality_constraint(DifferentiableMapPtr h);

  const util::MatrixSparsityPatern& hessian_sparcity_patern() const {
    return hessian_sparcity_patern_;
  }

 protected:
  DifferentiableMapPtr objective_function_;
  std::vector<DifferentiableMapPtr> inequality_constraints_;
  std::vector<DifferentiableMapPtr> equality_constraints_;

  uint32_t n_g_;  // number of inequality constraints.
  uint32_t n_h_;  // number of equality constraints.

  std::vector<util::MatrixSparsityPatern> g_gradient_sparcity_paterns_;
  std::vector<util::MatrixSparsityPatern> h_gradient_sparcity_paterns_;
  util::MatrixSparsityPatern hessian_sparcity_patern_;
};

}  // namespace numerical_optimization
}  // namespace bewego
