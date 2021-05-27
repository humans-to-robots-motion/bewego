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
 *                                               Jim Mainprice Wed 4 Feb 2020
 */
#pragma once

#include <bewego/motion/freeflyers.h>
#include <bewego/motion/trajectory.h>
#include <bewego/numerical_optimization/optimizer.h>
#include <bewego/numerical_optimization/trajectory_optimization.h>
#include <bewego/workspace/workspace.h>

namespace bewego {
namespace numerical_optimization {

class FreeflyerOptimzer : public TrajectoryOptimizer {
 public:
  FreeflyerOptimzer(uint32_t n,  // size of the configuration space
                    uint32_t T,  // number of cliques
                    double dt,   // time between cliques
                    ExtentBox workspace_bounds,
                    std::shared_ptr<Freeflyer> robot);

  /** @brief Adds a geodesic flow object to the optimizer */
  void set_geodesic_flow(DifferentiableMapPtr v) { geodesic_flow_ = v; }

  /** @brief Adds a geodesic flow object to the optimizer */
  void set_geodesic_distance(DifferentiableMapPtr v) { geodesic_distance_ = v; }

  /** @brief Adds a default configuration */
  void set_q_default(const Eigen::VectorXd& v) { q_default_ = v; }

  /** @brief Sets the end enffector of the freeflyer, this which what part of
   * the geometry should be used for setting up a goal attractor */
  void set_end_effector(uint32_t i) { end_effector_id_ = i; }

  /** @brief Adds a default configuration */
  void set_clique_collision_constraints(bool v) {
    clique_collision_constraints_ = v;
  }

  void AddGeodesicFlowTerm(double scalar);
  void AddGeodesicTerm(double scalar);
  void AddKeyPointsSurfaceConstraints(double margin, double scalar);
  void AddJointLimitConstraints(double scalar);
  void AddGoalConstraint(const Eigen::VectorXd& x_goal, double scalar);
  void AddPosturalTerms(double scalar);

 protected:
  /** @brief return bounds for dofs along the trajectory */
  std::vector<util::Bounds> DofsBounds() const;

  // Get ditance to obstacle
  ElementaryFunction GetDistanceActivation() const;

  // ipopt constraints
  std::vector<FunctionNetwork> GetKeyPointsSurfaceConstraints() const;

  // Workspace
  uint32_t workspace_dim_;                // Dimensionality of the workspace
  ExtentBox workspace_bounds_;            // Bounds of the workspace
  std::shared_ptr<Workspace> workspace_;  // Workspace geometry
  DifferentiableMapPtr smooth_collision_constraint_;

  // GeodesicFlow & Attractors
  DifferentiableMapPtr geodesic_flow_;
  DifferentiableMapPtr geodesic_distance_;
  std::string attractor_type_;
  double attractor_transition_;
  double attractor_interval_;
  bool attractor_value_geodesic_;
  bool attractor_make_smooth_;

  // Robot
  std::shared_ptr<Freeflyer> robot_;

  // End-effector
  uint32_t end_effector_id_;

  // Default posture
  Eigen::VectorXd q_default_;

  // Parameters
  double freeflyer_gamma_;
  double freeflyer_k_;
  bool clique_collision_constraints_;
};

}  // namespace numerical_optimization
}  // namespace bewego
