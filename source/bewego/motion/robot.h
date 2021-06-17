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
#pragma once

#include <bewego/derivatives/differentiable_map.h>
#include <bewego/motion/forward_kinematics.h>
#include <bewego/workspace/collision_checking.h>

namespace bewego {

/**
 * !\brief Differentiable wrapper around a kinematic chain
 */
class KinematicMap : public DifferentiableMap {
 public:
  KinematicMap() {
    type_ = "KinematicMap";
    PreAllocate();
  }
  KinematicMap(const std::string& name,                          // name
               std::shared_ptr<KinematicChain> kinematic_chain,  // kinematiscs
               uint32_t id_dof,                                  // id in kin.
               uint32_t id_frame_part                            // frame part
  );

  const std::string& name() const { return name_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const;
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const;

  uint32_t input_dimension() const { return kinematics_->nb_active_dofs(); }
  uint32_t output_dimension() const { return 3; }

 protected:
  std::string name_;
  uint32_t id_dof_;
  uint32_t id_frame_part_;
  std::shared_ptr<KinematicChain> kinematics_;
  mutable Eigen::VectorXd q_;
};

using KinematicMapPtr = std::shared_ptr<const KinematicMap>;

/**
 * !\brief TODO Add TaskMaps (keypoints) here.
 */
class Robot {
 public:
  Robot(std::shared_ptr<KinematicChain> kinematic_chain,
        const std::vector<std::pair<std::string, double>>& keypoints);

  Robot(const std::vector<RigidBodyInfo>& bodies,
        const std::vector<std::pair<std::string, double>>& keypoints);

  // returns the task map
  KinematicMapPtr task_map(std::string name) const {
    return task_maps_.at(name);
  }

  // Get Collision Points
  VectorOfCollisionPoints GetCollisionPoints() const;

  // returns the position task map of each keypoint
  std::vector<KinematicMapPtr> keypoint_task_maps() const {
    std::vector<KinematicMapPtr> task_maps(keypoints_.size());
    for (uint32_t i = 0; i < keypoints_.size(); i++) {
      task_maps[i] = task_map(keypoints_[i].first);
    }
    return task_maps;
  }

  // returns the keypoints radii
  std::vector<double> keypoint_radii() const {
    std::vector<double> radii(keypoints_.size());
    for (uint32_t i = 0; i < keypoints_.size(); i++) {
      radii[i] = keypoints_[i].second;
    }
    return radii;
  }

 protected:
  virtual void CreateTaskMaps();

  std::shared_ptr<KinematicChain> kinematic_chain_;        // FK
  std::map<std::string, KinematicMapPtr> task_maps_;       // element. taskmaps
  std::vector<std::pair<std::string, double>> keypoints_;  // keypoints
  std::map<std::string, double> keypoints_radii_;          // radii
};

}  // namespace bewego