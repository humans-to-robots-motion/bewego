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
 *                                               Jim Mainprice Mon 14 June 2021
 */

#include <bewego/motion/robot.h>

using namespace bewego;
// using namespace bewego::util;
using std::cerr;
using std::cout;
using std::endl;

//-----------------------------------------------------------------------------
// KinematicMap implementation.
//-----------------------------------------------------------------------------

KinematicMap::KinematicMap(
    const std::string& name,                          // name
    std::shared_ptr<KinematicChain> kinematic_chain,  // kinematiscs
    uint32_t id_dof,                                  // id in kin.
    uint32_t id_frame_part                            // frame part
    )
    : name_(name),
      kinematics_(kinematic_chain),
      id_dof_(id_dof),
      id_frame_part_(id_frame_part) {
  type_ = "KinematicMap";
  PreAllocate();
}

Eigen::VectorXd KinematicMap::Forward(const Eigen::VectorXd& q) const {
  if (q != q_) {
    kinematics_->SetAndUpdate(q);
    q_ = q;
  }
  return kinematics_->frame_part(id_dof_, id_frame_part_);
}

Eigen::MatrixXd KinematicMap::Jacobian(const Eigen::VectorXd& q) const {
  if (q != q_) {
    kinematics_->SetAndUpdate(q);
    q_ = q;
  }
  return kinematics_->JacobianFramePart(id_dof_, id_frame_part_);
}

//-----------------------------------------------------------------------------
// Robot implementation.
//-----------------------------------------------------------------------------

Robot::Robot(std::shared_ptr<KinematicChain> kinematic_chain,
             const std::vector<std::pair<std::string, double>>& keypoints)
    : kinematic_chain_(kinematic_chain), keypoints_(keypoints) {
  CreateTaskMaps();
}

Robot::Robot(const std::vector<RigidBodyInfo>& bodies,
             const std::vector<std::pair<std::string, double>>& keypoints)
    : kinematic_chain_(std::make_shared<KinematicChain>()),
      keypoints_(keypoints) {
  for (const auto& body : bodies) {
    kinematic_chain_->AddRigidBodyFromInfo(body);
  }
  CreateTaskMaps();
}

void Robot::CreateTaskMaps() {
  task_maps_.clear();
  keypoints_radii_.clear();
  std::string name;
  for (const auto& keypoint : keypoints_) {
    uint32_t id_dof = kinematic_chain_->rigid_body_id(keypoint.first);
    keypoints_radii_[keypoint.first] = keypoint.second;

    name = keypoint.first;
    task_maps_[name] = std::make_shared<KinematicMap>(  // Position map
        name, kinematic_chain_, id_dof, 0);

    name = keypoint.first + "_x";
    task_maps_[name] = std::make_shared<KinematicMap>(  // X axis map
        name, kinematic_chain_, id_dof, 1);

    name = keypoint.first + "_y";
    task_maps_[name] = std::make_shared<KinematicMap>(  // Y axis map
        name, kinematic_chain_, id_dof, 2);

    name = keypoint.first + "_z";
    task_maps_[name] = std::make_shared<KinematicMap>(  // Z axis map
        name, kinematic_chain_, id_dof, 3);
  }
}
