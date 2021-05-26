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
#include <bewego/numerical_optimization/optimizer.h>
#include <bewego/util/bounds.h>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>

namespace bewego {
namespace numerical_optimization {

class IpoptOptimizer : public ConstrainedOptimizer {
 public:
  IpoptOptimizer();
  virtual ~IpoptOptimizer() {}

  /*!\brief If a more general constrained optimization problem is passed in,
   * we try to downcast it to the required type. CHECK-fails if it is of the
   * wrong type.
   */
  ConstrainedSolution Run(const ConstrainedOptimizationProblem& problem,
                          const Eigen::VectorXd& x0) const override {
    try {
      return Run(
          dynamic_cast<const OptimizationProblemWithConstraints&>(problem), x0);
    } catch (const std::bad_cast& e) {
      std::cerr << "Interior point algorithm can be applied only to "
                << "OptimizationProblemWithConstraints types.";
    }
    return ConstrainedSolution();
  }

  /*!\brief Optimize problem with differentiable functions as input
   */
  ConstrainedSolution Run(const OptimizationProblemWithConstraints& problem,
                          const Eigen::VectorXd& x0) const;

  /*
   * !\brief get unbounded limits
   */
  std::vector<util::Bounds> GetVariableBounds(uint32_t n) const;
  std::vector<util::Bounds> GetGenericConstraintsBounds(uint32_t n_g,
                                                        uint32_t n_h) const;

  // Set bounds.
  void set_bounds(const std::vector<util::Bounds>& bounds) { bounds_ = bounds; }

  /*!\brief This function sets a value atouside the optimizer
   * for visualiztion and logging purpuses.
   * TODO test overhead and take out of verbose mode.
   * */
  void set_current_solution_accessor(
      std::function<void(const Eigen::VectorXd&)> f) {
    publish_solution_ = f;
  }

  /** Set options for the IPOPT solver. A complete list can be found here:
   * https://www.coin-or.org/Ipopt/documentation/node40.html
   */
  void set_option(const std::string& name, const std::string& value);
  void set_option(const std::string& name, int value);
  void set_option(const std::string& name, double value);

  // set options directly from map
  void set_options_map(const std::map<std::string, double>& options);

  /** @brief  Get the total wall clock time for the optimization, including
   * function evaluations.
   */
  double total_wallclock_time() const;

 private:
  std::shared_ptr<Ipopt::IpoptApplication> ipopt_app_;
  std::vector<util::Bounds> bounds_;
  mutable std::atomic_bool preempt_;
  std::function<void(const Eigen::VectorXd&)> publish_solution_;
};

}  // namespace numerical_optimization
}  // namespace bewego
