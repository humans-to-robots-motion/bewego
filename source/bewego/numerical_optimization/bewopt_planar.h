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

#include <bewego/numerical_optimization/trajectory_optimization.h>
#include <bewego/workspace/workspace.h>

namespace bewego {
namespace numerical_optimization {

class PlanarOptimizer : public TrajectoryOptimizer {
 public:
  PlanarOptimizer(uint32_t T,  // number of cliques
                  double dt,   // time between cliques
                  const std::vector<double>& workspace_bounds  // bounds
  );

  /** @brief Adds collision constraints with the environment */
  void AddKeyPointsSurfaceConstraints(double margin, double scalar);

  /** @brief Adds collision constraints with the environment */
  void AddSmoothKeyPointsSurfaceConstraints(double scalar);

  /** @brief Adds goal constraint */
  void AddInequalityConstraintToEachActiveClique(DifferentiableMapPtr phi,
                                                 double scalar);

  /** @brief Adds goal constraint */
  void AddGoalConstraint(const Eigen::VectorXd& q_goal, double scalar);

  /** @brief Adds goal on circle constraint */
  void AddGoalManifoldConstraint(const Eigen::VectorXd& q_goal, double radius,
                                 double scalar);

  /** @brief Adds waypoint constraint */
  void AddWayPointConstraint(const Eigen::VectorXd& q_waypoint, uint32_t t,
                             double scalar);

  /** @brief Adds waypoint manifold constraint */
  void AddWayPointManifoldConstraint(const Eigen::VectorXd& q_waypoint,
                                     uint32_t t, double radius, double scalar);

  /** @brief Returns the bounds for one dof */
  std::vector<Bounds> DofsBounds() const {
    assert(n_ == 2);
    std::vector<Bounds> limits(n_);
    auto extent = workspace_bounds_->extent();
    limits[0] = Bounds(extent.x_min(), extent.x_max());
    limits[1] = Bounds(extent.y_min(), extent.y_max());
    return limits;
  }

 protected:
  // Bounds of the workspace
  std::shared_ptr<Rectangle> workspace_bounds_;
};

}  // namespace numerical_optimization
}  // namespace bewego
