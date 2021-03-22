/*
 * Copyright (c) 2020
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
 *                                               Jim Mainprice Wed 10 Mar 2021
 */
#pragma once

#include <bewego/motion/objective.h>
#include <bewego/motion/publisher.h>
#include <bewego/motion/trajectory.h>
#include <bewego/numerical_optimization/optimizer.h>
#include <bewego/numerical_optimization/trajectory_optimization.h>
#include <bewego/workspace/workspace.h>

namespace bewego {
namespace numerical_optimization {

class PlanarOptimizer : public MotionObjective {
 public:
  PlanarOptimizer(uint32_t T,  // number of cliques
                  double dt,   // time between cliques
                  const std::vector<double>& workspace_bounds  // bounds
  );

  /** @brief Adds collision constraints with the environment */
  void AddKeyPointsSurfaceConstraints(double margin, double scalar);

  /** @brief Adds collision constraints with the environment */
  void AddSmoothKeyPointsSurfaceConstraints(double margin, double gamma,
                                            double scalar);

  /** @brief Adds goal constraint */
  void AddInequalityConstraintToEachActiveClique(DifferentiableMapPtr phi,
                                                 double scalar);

  /** @brief Adds goal constraint */
  void AddGoalConstraint(const Eigen::VectorXd& q_goal, double scalar);

  /** @brief Optimize a given trajectory */
  OptimizeResult Optimize(
      const Eigen::VectorXd& initial_traj,          // entire trajectory
      const Eigen::VectorXd& x_goal,                // goal configuration
      const std::map<std::string, double>& options  // optimizer options
  ) const;

  // @brief Adds trajectory publisher (t_pause in microseconds)
  void set_trajectory_publisher(bool with_slow_down, uint32_t t_pause = 100000);

 protected:
  typedef CliquesFunctionNetwork FunctionNetwork;
  typedef std::shared_ptr<const FunctionNetwork> FunctionNetworkPtr;
  typedef std::shared_ptr<const DifferentiableMap> ElementaryFunction;

  std::shared_ptr<const ConstrainedOptimizer> SetupIpoptOptimizer(
      const Eigen::VectorXd& q_init,
      const std::map<std::string, double>& ipopt_options) const;

  std::vector<Bounds> DofBounds() const;            // Dof bounds limits
  std::vector<Bounds> TrajectoryDofBounds() const;  // Dof bounds trajectory

  // Constraints networks
  std::vector<FunctionNetworkPtr> g_constraints_;  // inequalities
  std::vector<FunctionNetworkPtr> h_constraints_;  // equalities

  // Bounds of the workspace
  std::shared_ptr<Rectangle> workspace_bounds_;

  // options
  bool with_rotation_;
  bool with_attractor_constraint_;
  bool ipopt_with_bounds_;
  std::string ipopt_hessian_approximation_;

  // Logging
  bool visualize_inner_loop_;
  bool visualize_slow_down_;
  uint32_t visualize_t_pause_;
  bool monitor_inner_statistics_;
  mutable std::shared_ptr<TrajectoryPublisher> publisher_;
  // visualizer_; mutable std::shared_ptr<util::StatsMonitor> stats_monitor_;
};

}  // namespace numerical_optimization
}  // namespace bewego
