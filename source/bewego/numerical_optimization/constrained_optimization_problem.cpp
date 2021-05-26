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

#include <bewego/numerical_optimization/constrained_optimization_problem.h>
#include <bewego/util/misc.h>

using namespace bewego;
using namespace bewego::util;
using namespace numerical_optimization;
using std::cout;
using std::endl;

//------------------------------------------------------------------------------
// Simple optimization implementation
//------------------------------------------------------------------------------

OptimizationProblemWithConstraints::OptimizationProblemWithConstraints(
    DifferentiableMapPtr objective_function,
    const std::vector<DifferentiableMapPtr>& inequality_constraints,
    const std::vector<DifferentiableMapPtr>& equality_constraints)
    : n_g_(0), n_h_(0) {
  // Store all functions
  objective_function_ = objective_function;
  inequality_constraints_ = inequality_constraints;
  equality_constraints_ = equality_constraints;

  // Make sure all functions have the same size.
  for (auto& eq : equality_constraints_) {
    assert(objective_function_->input_dimension() == eq->input_dimension());
  }
  for (auto& neq : inequality_constraints_) {
    assert(objective_function_->input_dimension() == neq->input_dimension());
  }

  // Check that sizes don't not overflow
  n_g_ = size_t_to_uint(inequality_constraints_.size());
  n_h_ = size_t_to_uint(equality_constraints_.size());
}

OptimizationProblemWithConstraints::~OptimizationProblemWithConstraints() {}

void OptimizationProblemWithConstraints::add_inequality_constraint(
    DifferentiableMapPtr g) {
  assert(objective_function_);
  assert(objective_function_->input_dimension() == g->input_dimension());
  inequality_constraints_.push_back(g);
  n_g_++;
}

void OptimizationProblemWithConstraints::add_equality_constraint(
    DifferentiableMapPtr h) {
  assert(objective_function_);
  assert(objective_function_->input_dimension() == h->input_dimension());
  equality_constraints_.push_back(h);
  n_h_++;
}

double OptimizationProblemWithConstraints::Evaluate(
    const Eigen::VectorXd& x, Eigen::VectorXd* g_evaluations,
    Eigen::VectorXd* h_evaluations) const {
  assert(g_evaluations != nullptr);
  assert(h_evaluations != nullptr);

  *g_evaluations = Eigen::VectorXd(num_inequality_constraints());
  for (uint32_t i = 0; i < num_inequality_constraints(); i++) {
    (*g_evaluations)[i] = inequality_constraints_[i]->ForwardFunc(x);
  }
  *h_evaluations = Eigen::VectorXd(num_equality_constraints());
  for (uint32_t i = 0; i < num_equality_constraints(); i++) {
    (*h_evaluations)[i] = equality_constraints_[i]->ForwardFunc(x);
  }
  return objective_function_->ForwardFunc(x);
}
