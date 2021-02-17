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
#include <bewego/cost_terms.h>
#include <bewego/objective.h>

namespace bewego {

void MotionObjective::AddSmoothnessTerms(uint32_t deriv_order, double scalar) {
  if (deriv_order == 1) {
    auto derivative = std::make_shared<Compose>(
        std::make_shared<SquaredNormVelocity>(config_space_dim_, dt_),
        function_network_->LeftOfCliqueMap());
    function_network_->RegisterFunctionForAllCliques(
        std::make_shared<Scale>(derivative, scalar));
  } else if (deriv_order == 2) {
    auto derivative =
        std::make_shared<SquaredNormAcceleration>(config_space_dim_, dt_);
    function_network_->RegisterFunctionForAllCliques(
        std::make_shared<Scale>(derivative, scalar));
  } else {
    std::cerr << "deriv_order (" << deriv_order << ") not suported"
              << std::endl;
  }
}

void MotionObjective::AddIsometricPotentialToAllCliques(
    DifferentiableMapPtr potential, double scalar) {
  auto cost = std::make_shared<Compose>(potential, 
      function_network_->CenterOfCliqueMap());

  auto squared_norm_vel = std::make_shared<Compose>(
      std::make_shared<SquaredNormVelocity>(config_space_dim_, dt_),
      function_network_->RightOfCliqueMap());

  function_network_->RegisterFunctionForAllCliques(std::make_shared<Scale>(
      std::make_shared<ProductMap>(cost, squared_norm_vel), scalar));
}

void MotionObjective::AddObstacleTerms(double scalar, double alpha,
                                       double margin) {
  auto sdf = workspace_->SignedDistanceField();
  obstacle_potential_ = std::make_shared<ObstaclePotential>(sdf, alpha, margin);
  AddIsometricPotentialToAllCliques(obstacle_potential_, scalar);
}

void MotionObjective::AddTerminalPotentialTerms(const Eigen::VectorXd& q_goal,
                                                double scalar) {
  auto terminal_potential =
      std::make_shared<Compose>(std::make_shared<SquaredNorm>(q_goal),
                                function_network_->CenterOfCliqueMap());

  function_network_->RegisterFunctionForLastClique(
      std::make_shared<Scale>(terminal_potential, scalar));
}

void MotionObjective::AddWayPointTerms(const Eigen::VectorXd& q_waypoint,
                                       uint32_t t, double scalar) {
  auto potential =
      std::make_shared<Compose>(std::make_shared<SquaredNorm>(q_waypoint),
                                function_network_->LeftMostOfCliqueMap());

  function_network_->RegisterFunctionForClique(
      t, std::make_shared<Scale>(potential, scalar));
}

void MotionObjective::AddSphere(const Eigen::VectorXd& center, double radius) {
  workspace_objects_.push_back(std::make_shared<Circle>(center, radius));
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
}

void MotionObjective::AddBox(const Eigen::VectorXd& center,
                             const Eigen::VectorXd& dimension) {
  workspace_objects_.push_back(
      std::make_shared<Rectangle>(center, dimension, 0));
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
}

void MotionObjective::ClearWorkspace() {
  workspace_objects_.clear();
  workspace_ = std::make_shared<Workspace>(workspace_objects_);
}

}  // namespace bewego