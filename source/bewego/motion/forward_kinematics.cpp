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
#include <bewego/motion/forward_kinematics.h>

using std::cout;
using std::endl;

namespace bewego {

//-----------------------------------------------------------------------------
// RigidBody function implementation.
//-----------------------------------------------------------------------------

RigidBody::RigidBody(const std::string& name, const std::string& joint_name,
                     uint32_t joint_type_id, double dof_lower_limit,
                     double dof_upper_limit,
                     const Eigen::Affine3d& local_in_prev,
                     const Eigen::Vector3d& joint_axis_in_local)
    : debug_(false),
      name_(name),
      joint_bounds_(dof_lower_limit, dof_upper_limit),
      joint_name_(joint_name),
      joint_type(JointType(joint_type_id)),
      frame_in_base_(Eigen::Affine3d::Identity()),
      frame_in_local_(Eigen::Affine3d::Identity()),
      local_in_prev_(local_in_prev),
      joint_axis_in_local_(joint_axis_in_local),
      joint_axis_in_base_(Eigen::Vector3d::Zero()) {
  if (debug_) {
    cout << "***** Create Rigid body ** " << endl;
    cout << " -- joint_name_  : " << joint_name_ << endl;
    cout << " -- joint_type  : " << joint_type << endl;
    cout << " -- local_in_prev_  : " << endl << local_in_prev_.matrix() << endl;
    cout << " -- joint_axis_in_local_  : " << joint_axis_in_local_.transpose()
         << endl;
  }
}

//-----------------------------------------------------------------------------
// KinematicChain function implementation.
//-----------------------------------------------------------------------------

Eigen::MatrixXd KinematicChain::JacobianPosition(int link_index) const {
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, rigid_bodies_.size());
  auto x = rigid_bodies_[link_index]->joint_origin_in_base();
  for (int j = 0; j < J.cols(); j++) {
    const auto& joint_origin = rigid_bodies_[j]->joint_origin_in_base();
    const auto& joint_axis = rigid_bodies_[j]->joint_axis_in_base();
    J.col(j) = joint_axis.cross(x - joint_origin);
    if (link_index == j) break;
  }
  return J;
}

Eigen::MatrixXd KinematicChain::JacobianAxis(int link_index,
                                             int axis_index) const {
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, rigid_bodies_.size());
  const Eigen::Vector3d& link_axis =
      rigid_bodies_[link_index]->frame_in_base().linear().col(axis_index);
  for (int j = 0; j < J.cols(); j++) {
    const auto& joint_origin = rigid_bodies_[j]->joint_origin_in_base();
    const auto& joint_axis = rigid_bodies_[j]->joint_axis_in_base();
    J.col(j) = joint_axis.cross(link_axis);
    if (link_index == j) break;
  }

  return J;
}

Eigen::MatrixXd KinematicChain::JacobianFrame(int link_index) const {
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3 * 4, rigid_bodies_.size());
  auto x = rigid_bodies_[link_index]->joint_origin_in_base();
  const auto& link_rotation_in_base =
      rigid_bodies_[link_index]->frame_in_base().linear();
  for (int j = 0; j < J.cols(); j++) {
    const auto& joint_origin = rigid_bodies_[j]->joint_origin_in_base();
    const auto& joint_axis = rigid_bodies_[j]->joint_axis_in_base();

    // Jacobian of Position
    J.col(j).head(3) = joint_axis.cross(x - joint_origin);

    // Jacobians of Axes
    for (int k = 0; k < 3; ++k) {
      const auto& link_axis = link_rotation_in_base.col(k);
      J.col(j).segment((k + 1) * 3, 3) = joint_axis.cross(link_axis);
    }
    if (link_index == j) break;
  }
  return J;
}

}  // namespace bewego