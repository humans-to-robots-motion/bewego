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

#include <bewego/derivatives/differentiable_map.h>
#include <bewego/motion/cost_terms.h>
#include <bewego/motion/trajectory.h>
#include <bewego/workspace/softmax_primitive_workspace.h>
#include <bewego/workspace/workspace.h>

#include <iostream>

namespace bewego {

class MotionObjective {
 public:
  MotionObjective(uint32_t T, double dt, uint32_t config_space_dim);

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
  void AddIsometricPotentialToClique(DifferentiableMapPtr potential, uint32_t t,
                                     double scalar);

  /** Takes a matrix and adds an isometric potential term to all clique */
  virtual void AddObstacleTerms(double scalar, double alpha);

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

  /** Reconstruct Workspace and Obstacle fields **/
  void ReconstructWorkspace();

  /** Removes the objects from the workspace */
  void ClearWorkspace();

  /** Set the distance field smoothness */
  void SetSDFGamma(double v) {
    gamma_ = v;
    ReconstructWorkspace();
  }

  /** Set the distance field margin */
  void SetSDFMargin(double v) {
    obstacle_margin_ = v;
    ReconstructWorkspace();
  }

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

  void set_verbose(bool v) { verbose_ = v; }

 protected:
  bool verbose_;
  double T_;                // Number of active cliques
  double dt_;               // time interval between cliques
  double n_;                // Dimensionality of the configuration space
  double gamma_;            // Smoothing parameter of the SDF
  double obstacle_margin_;  // Margin parameter of the SDF
  std::shared_ptr<CliquesFunctionNetwork> function_network_;
  std::shared_ptr<Workspace> workspace_;
  std::vector<std::shared_ptr<const WorkspaceObject>> workspace_objects_;
  std::shared_ptr<ObstaclePotential> obstacle_potential_;
  std::vector<DifferentiableMapPtr> smooth_sdf_;
};

}  // namespace bewego