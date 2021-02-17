/*
 * Copyright (c) 2021
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
 *                                                             Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <bewego/differentiable_map.h>
#include <bewego/trajectory.h>
#include <bewego/workspace.h>
#include <bewego/cost_terms.h>

#include <iostream>

namespace bewego {

class MotionObjective {
 public:
  MotionObjective(uint32_t T, double dt, uint32_t config_space_dim)
      : T_(T), dt_(dt), config_space_dim_(config_space_dim) {
    assert(T_ > 2);
    assert(dt_ > 0);
    assert(config_space_dim_ > 0);
    function_network_ = std::make_shared<CliquesFunctionNetwork>(
        (T_ + 2) * config_space_dim_, config_space_dim_);
    workspace_objects_.clear();
    workspace_ = std::make_shared<Workspace>(workspace_objects_);
    ClearWorkspace();
  }

  /** Apply the following euqation to all cliques:

              | d^n/dt^n x_t |^2

      where n (deriv_order) is either
        1: velocity
        2: accleration
  **/
  void AddSmoothnessTerms(uint32_t deriv_order, double scalar);

  /** Apply the following euqation to all cliques:

              c(x_t) | d/dt x_t |

        The resulting Riemanian metric is isometric. TODO see paper.
        Introduced in CHOMP, Ratliff et al. 2009.
  **/
  void AddIsometricPotentialToAllCliques(DifferentiableMapPtr potential,
                                         double scalar);

  /** Takes a matrix and adds an isometric potential term to all clique */
  void AddObstacleTerms(double scalar, double alpha, double margin);

  /** Add terminal potential

            phi(x) =  | q_T - q_goal |^2
  **/
  void AddTerminalPotentialTerms(const Eigen::VectorXd& q_goal, double scalar);

  /** Add waypoint

            phi(x) =  | q_t - q_waypoint |^2
  **/
  void AddWayPointTerms(const Eigen::VectorXd& q_waypoint, uint32_t t,
                        double scalar);

  /** Add Sphere to Workspace (2D for now) */
  void AddSphere(const Eigen::VectorXd& center, double radius);

  /** Add Box to Workspace (2D for now) */
  void AddBox(const Eigen::VectorXd& center, const Eigen::VectorXd& dimension);

  /** Removes the objects from the workspace */
  void ClearWorkspace();

  std::shared_ptr<const CliquesFunctionNetwork> function_network() const {
    return function_network_;
  }

  std::shared_ptr<const TrajectoryObjectiveFunction> objective(
      const Eigen::VectorXd& q_init) const {
    return std::make_shared<TrajectoryObjectiveFunction>(q_init,
                                                         function_network_);
  }

  std::shared_ptr<const ObstaclePotential> obstacle_potential() const {
    return obstacle_potential_;
  }

 protected:
  double T_;
  double dt_;
  double config_space_dim_;
  std::shared_ptr<CliquesFunctionNetwork> function_network_;
  std::shared_ptr<Workspace> workspace_;
  std::vector<std::shared_ptr<const WorkspaceObject>> workspace_objects_;
  std::shared_ptr<ObstaclePotential> obstacle_potential_;
};

}  // namespace bewego