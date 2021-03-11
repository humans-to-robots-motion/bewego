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

namespace bewego {

void MotionObjective::AddSmoothnessTerms(uint32_t deriv_order, double scalar) {
  if (deriv_order == 1) {
    auto derivative =
        ComposedWith(std::make_shared<SquaredNormVelocity>(n_, dt_),
                     function_network_->LeftOfCliqueMap());
    function_network_->RegisterFunctionForAllCliques(scalar * derivative);
  } else if (deriv_order == 2) {
    auto derivative = std::make_shared<SquaredNormAcceleration>(n_, dt_);
    function_network_->RegisterFunctionForAllCliques(scalar * derivative);
  } else {
    std::cerr << "deriv_order (" << deriv_order << ") not suported"
              << std::endl;
  }
}

void MotionObjective::AddIsometricPotentialToAllCliques(
    DifferentiableMapPtr potential, double scalar) {
  auto cost = ComposedWith(potential, function_network_->CenterOfCliqueMap());
  auto squared_velocity_norm =
      ComposedWith(potential, function_network_->CenterOfCliqueMap());
  auto c_v_norm = cost * squared_velocity_norm;
  function_network_->RegisterFunctionForAllCliques(scalar * c_v_norm);
}

void MotionObjective::AddObstacleTerms(double scalar, double alpha,
                                       double margin) {
  auto sdf = workspace_->SignedDistanceField();
  obstacle_potential_ = std::make_shared<ObstaclePotential>(sdf, alpha, 1);
  AddIsometricPotentialToAllCliques(obstacle_potential_, scalar);
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
  auto potential = ComposedWith(std::make_shared<SquaredNorm>(q_waypoint),
                                function_network_->LeftMostOfCliqueMap());
  function_network_->RegisterFunctionForClique(t, scalar * potential);
}

void MotionObjective::AddSphere(const Eigen::VectorXd& center, double radius) {
  workspace_objects_.push_back(std::make_shared<Circle>(center, radius));
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
  auto sdf = workspace_->SignedDistanceField();
  obstacle_potential_ = std::make_shared<ObstaclePotential>(sdf, 10, 1);
}

void MotionObjective::AddBox(const Eigen::VectorXd& center,
                             const Eigen::VectorXd& dimension) {
  workspace_objects_.push_back(
      std::make_shared<Rectangle>(center, dimension, 0));
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
  auto sdf = workspace_->SignedDistanceField();
  obstacle_potential_ = std::make_shared<ObstaclePotential>(sdf, 10, 1);
}

void MotionObjective::ClearWorkspace() {
  workspace_objects_.clear();
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
}

}  // namespace bewego