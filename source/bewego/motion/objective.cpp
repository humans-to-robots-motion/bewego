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
#include <bewego/motion/cost_terms.h>
#include <bewego/motion/objective.h>
#include <bewego/workspace/collision_checking.h>

using std::cerr;
using std::cout;
using std::endl;

namespace bewego {

MotionObjective::MotionObjective(uint32_t T, double dt,
                                 uint32_t config_space_dim)
    : verbose_(false), T_(T), dt_(dt), n_(config_space_dim) {
  assert(T_ > 2);
  assert(dt_ > 0);
  assert(n_ > 0);
  cout << "T_ in MotionObjective : " << T_ << endl;
  function_network_ =
      std::make_shared<CliquesFunctionNetwork>((T_ + 2) * n_, n_);
  gamma_ = 40;
  obstacle_margin_ = 0;
  cout << "Clear workspace..." << endl;
  ClearWorkspace();
}

void MotionObjective::AddSmoothnessTerms(uint32_t deriv_order, double scalar) {
  DifferentiableMapPtr derivative;
  if (deriv_order == 1) {
    derivative = SquaredVelocityNorm(n_, dt_);
    derivative = ComposedWith(derivative, function_network_->LeftOfCliqueMap());
  } else if (deriv_order == 2) {
    derivative = SquaredAccelerationNorm(n_, dt_);
  } else {
    cerr << "WARNING: deriv_order (" << deriv_order << ") not suported" << endl;
    return;
  }
  function_network_->RegisterFunctionForAllCliques(dt_ * scalar * derivative);
}

void MotionObjective::AddIsometricPotentialToClique(
    DifferentiableMapPtr potential, uint32_t t, double scalar) {
  auto center_clique = function_network_->CenterOfCliqueMap();
  auto right_clique = function_network_->RightOfCliqueMap();
  auto sq_norm_vel = ComposedWith(SquaredVelocityNorm(n_, dt_), right_clique);
  auto phi = ComposedWith(potential, center_clique);
  function_network_->RegisterFunctionForClique(
      t, dt_ * scalar * (phi * sq_norm_vel));
}

void MotionObjective::AddObstacleTerms(double scalar, double alpha) {
  if (workspace_objects_.empty()) {
    cerr << "WARNING: no obstacles are in the workspace" << endl;
    return;
  }
  // we use a vector of smooth distance to efficiently use cache
  for (uint32_t t = 0; t < function_network_->nb_cliques(); t++) {
    auto obstacle_potential =
        std::make_shared<ObstaclePotential>(smooth_sdf_[t], alpha, 1);
    AddIsometricPotentialToClique(obstacle_potential, t, scalar);
  }
}

void MotionObjective::AddTerminalPotentialTerms(const Eigen::VectorXd& q_goal,
                                                double scalar) {
  auto terminal_potential =
      ComposedWith(std::make_shared<SquaredNorm>(q_goal),
                   function_network_->CenterOfCliqueMap());
  function_network_->RegisterFunctionForLastClique(scalar * terminal_potential);
}

void MotionObjective::AddWayPointTerms(const Eigen::VectorXd& q_waypoint,
                                       uint32_t t, double scalar) {
  auto d_waypoint = std::make_shared<SquaredNorm>(q_waypoint);
  auto phi = ComposedWith(d_waypoint, function_network_->LeftMostOfCliqueMap());
  function_network_->RegisterFunctionForClique(t, scalar * phi);
}

void MotionObjective::AddSphere(const Eigen::VectorXd& center, double radius) {
  workspace_objects_.push_back(std::make_shared<Circle>(center, radius));
  ReconstructWorkspace();
}

void MotionObjective::AddBox(const Eigen::VectorXd& center,
                             const Eigen::VectorXd& dimension) {
  workspace_objects_.push_back(
      std::make_shared<Rectangle>(center, dimension, 0));
  ReconstructWorkspace();
}

void MotionObjective::ReconstructWorkspace() {
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
  auto sdf = workspace_->SignedDistanceField();
  obstacle_potential_ = std::make_shared<ObstaclePotential>(sdf, 10, 1);
  auto surfaces = workspace_->ExtractSurfaceFunctions();
  smooth_sdf_.clear();
  for (uint32_t t = 0; t < function_network_->nb_cliques(); t++) {
    smooth_sdf_.push_back(std::make_shared<SmoothCollisionConstraint>(
        surfaces, gamma_, obstacle_margin_));
  }
}

void MotionObjective::ClearWorkspace() {
  workspace_objects_.clear();
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
}

}  // namespace bewego