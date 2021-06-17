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
 *                                                              Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;

namespace bewego {

/*!\brief Represents bounds \in [lower, upper].
 * for a scalar variable
 */
class ScalarBound {
 public:
  ScalarBound() {}

  ScalarBound(double lower, double upper) : lower_(lower), upper_(upper) {}

  ScalarBound(const std::pair<double, double>& bounds)
      : lower_(bounds.first), upper_(bounds.second) {}

  double lower() const { return lower_; }
  double upper() const { return upper_; }
  double midpoint() const { return .5 * (upper_ + lower_); }

  void set_lower(double value) { lower_ = value; }
  void set_upper(double value) { upper_ = value; }

  /*!\brief Returns true if variable is within bounds.
   */
  bool IsWithinBounds(double x) const { return lower_ <= x && x <= upper_; }

 private:
  double lower_;
  double upper_;
};

/*!\brief Represents a rigid body
 *
 * A rigid body has one Degree of Freedom (DoF) associated to it
 * it can be rotational or translational around or along
 * an axis called the "joint axis". A local frame is defined
 * that is confound to the local_in_prev when the DoF is zero.
 * When the DoF is set, the RigdBody's frame in local coordinates
 * is updated. The frame of the rigid body in global coordinates
 * can be computed and stored when "Propagate" is called.
 * There, the DoF of the rigid body is offset by
 * the DoFs of other rigid bodies if it is part of a chain or tree.
 */
class RigidBody {
 public:
  enum JointType { ROTATIONAL = 0, TRANSLATIONAL = 1, FIXED = 4 } joint_type;

  RigidBody() {}
  RigidBody(const std::string& name, const std::string& joint_name,
            uint32_t joint_type_id, double dof_lower_limit,
            double dof_upper_limit, const Eigen::Affine3d& local_in_prev,
            const Eigen::Vector3d& joint_axis_in_local);

  /*!\brief Update this rigid body's post joint
   * transform (i.e., (frame_in_local_))
   */
  void SetDoF(double v) {
    switch (joint_type) {
      case ROTATIONAL:
        frame_in_local_.linear() =
            Eigen::AngleAxisd(v, joint_axis_in_local_).toRotationMatrix();
        break;
      case TRANSLATIONAL:
        frame_in_local_.translation() = v * joint_axis_in_local_;
        break;
      case FIXED:
        // Do nothing.
        break;
      default:
        std::cerr << "This joint type is not supported: " << joint_type;
    }
  }

  /*!\brief Calculate forward kinematics
   * T_prev is the previous rigid body frame
   */
  const Eigen::Affine3d& Propagate(const Eigen::Affine3d& T_prev) {
    if (joint_type == FIXED) {
      frame_in_base_ = T_prev * local_in_prev_;
    } else {
      frame_in_base_ = T_prev * local_in_prev_ * frame_in_local_;
      joint_axis_in_base_ = frame_in_base_.linear() * joint_axis_in_local_;
    }
    return frame_in_base_;
  }

  /*!\brief Frame accessor.
   */
  const std::string& name() const { return name_; }
  const std::string& joint_name() const { return joint_name_; }
  const Eigen::Affine3d& frame_in_base() const { return frame_in_base_; }
  const Eigen::Affine3d& local_in_prev() const { return local_in_prev_; }
  const Eigen::Vector3d& joint_axis_in_base() const {
    return joint_axis_in_base_;
  }
  Eigen::Vector3d joint_origin_in_base() const {
    return frame_in_base_.translation();
  }

 protected:
  bool debug_;                // Debug flag
  std::string name_;          // Rigid body name in URDF.
  ScalarBound joint_bounds_;  // Parent joint limits
  std::string joint_name_;    // Joint body name

  Eigen::Affine3d frame_in_base_;   // FK solution
  Eigen::Affine3d frame_in_local_;  // Joint transformed frame
  Eigen::Affine3d local_in_prev_;   // Prev to curr transform

  // Info about the joint axis.
  // If it's axis aligned, then we can skip some computation.
  Eigen::Vector3d joint_axis_in_local_;
  Eigen::Vector3d joint_axis_in_base_;
};

/**
 * !\brief Represents the info necessary to represent
 * a rigid body. This can be filled form python, loaded
 * by parsing a URDF file or something else.
 */
class RigidBodyInfo {
 public:
  RigidBodyInfo() {}
  RigidBodyInfo(const RigidBodyInfo& m) { *this = m; }
  std::string name;
  std::string joint_name;
  uint32_t joint_type;
  double dof_lower_limit;
  double dof_upper_limit;
  Eigen::Matrix4d local_in_prev;
  Eigen::Vector3d joint_axis_in_local;
};

/**
 * !\brief Represents a chain of rigid bodies
 *
 * TODO:
 *      - extend beyond chain to handle trees by combing chains.
 *      - If we then derive from a single Kinematic class chains and trees
 *        we should be able to integrate trees seamlessly
 */
class KinematicChain {
 public:
  KinematicChain() {
    base_ = Eigen::Isometry3d::Identity();
    rigid_bodies_.clear();
    active_dofs_.clear();
  }

  KinematicChain(const std::vector<RigidBodyInfo>& rigid_bodies_info,
                 const Eigen::Isometry3d& base = Eigen::Isometry3d::Identity())
      : base_(base) {
    rigid_bodies_.clear();
    active_dofs_.clear();
    for (const auto& info : rigid_bodies_info) {
      AddRigidBodyFromInfo(info);
    }
  }

  virtual ~KinematicChain();

  void AddRigidBodyFromInfo(const RigidBodyInfo& info) {
    AddRigidBody(info.name,                // name of the RB
                 info.joint_name,          // name of the joint
                 info.joint_type,          // ROT : 0, TRANS : 1, FIXED = 4
                 info.dof_lower_limit,     // DoF upper limit
                 info.dof_upper_limit,     // DoF lower limit
                 info.local_in_prev,       // Transform from prev
                 info.joint_axis_in_local  // joint axis
    );
  }

  void AddRigidBody(const std::string& name, const std::string& joint_name,
                    uint32_t joint_type, double dof_lower_limit,
                    double dof_upper_limit,
                    const Eigen::Matrix4d& local_in_prev,
                    const Eigen::Vector3d& joint_axis_in_local) {
    Eigen::Affine3d t;
    t.matrix() = local_in_prev;

    rigid_bodies_.push_back(std::make_shared<RigidBody>(
        name, joint_name, joint_type, dof_lower_limit, dof_upper_limit, t,
        joint_axis_in_local));

    if (joint_type != RigidBody::JointType::FIXED) {
      active_dofs_.push_back(rigid_bodies_.back());
    }
  }

  void SetAndUpdate(const Eigen::VectorXd& q) {
    SetConfiguration(q);
    ForwardKinematics();
  }

  void SetConfiguration(const Eigen::VectorXd& q) {
    if (active_dofs_.size() != q.size()) {
      throw std::runtime_error("KinematicChain : input dimension missmatch " +
                               std::to_string(active_dofs_.size()) + ", " +
                               std::to_string(q.size()));
    }
    for (uint32_t i = 0; i < active_dofs_.size(); i++) {
      active_dofs_[i]->SetDoF(q[i]);
    }
  }

  // Sets the transform from base
  void set_base_transform(const Eigen::Matrix4d& t) { base_.matrix() = t; }

  /**
   * Performs forward kinematics by propagating
   * the affine transformations through the chain
   */
  void ForwardKinematics() {
    rigid_bodies_[0]->Propagate(base_);
    for (uint32_t i = 1; i < rigid_bodies_.size(); i++) {
      auto& child = rigid_bodies_[i];
      auto& parent = rigid_bodies_[i - 1];
      child->Propagate(parent->frame_in_base());
    }
  }

  // Assumes that Forward Kinematics has been called.
  Eigen::MatrixXd JacobianPosition(int link_index) const;

  // Assumes that Forward Kinematics has been called.
  // { 0 : x axis,  1 : y axis,  2 : z axis }
  Eigen::MatrixXd JacobianAxis(int link_index, int axis_index) const;

  // Assumes that Forward Kinematics has been called.
  Eigen::MatrixXd JacobianFrame(int link_index) const;

  // Assumes that Forward Kinematics has been called.
  Eigen::Vector3d position(uint32_t idx) const {
    return rigid_bodies_[idx]->frame_in_base().translation();
  }

  // Assumes that Forward Kinematics has been called.
  // { 0 : x axis,  1 : y axis,  2 : z axis }
  Eigen::Vector3d axis(uint32_t idx, uint32_t axis_index) const {
    return rigid_bodies_[idx]->frame_in_base().linear().col(axis_index);
  }

  // Assumes that Forward Kinematics has been called.
  Eigen::Matrix3d rotation(uint32_t idx) const {
    return rigid_bodies_[idx]->frame_in_base().linear();
  }

  // Assumes that Forward Kinematics has been called.
  const Eigen::Matrix4d& transform(uint32_t idx) const {
    return rigid_bodies_[idx]->frame_in_base().matrix();
  }

  // Throws an exception if it is not consistent
  void CheckConsistentQuerry(uint32_t idx, uint32_t id_part) const;

  /**
   * Returns the jacobian of part of a frame
   * Assumes that Forward Kinematics has been called.
   * id_part = 0 : position
   * id_part = 1 : x axis
   * id_part = 2 : y axis
   * id_part = 3 : z axis
   */
  Eigen::MatrixXd JacobianFramePart(uint32_t idx, uint32_t id_part) const {
    CheckConsistentQuerry(idx, id_part);
    return id_part == 0 ? JacobianPosition(idx)
                        : JacobianAxis(idx, id_part - 1);
  }

  /**
   * Returns a part of the frame
   * Assumes that Forward Kinematics has been called.
   * id_part = 0 : position
   * id_part = 1 : x axis
   * id_part = 2 : y axis
   * id_part = 3 : z axis
   */
  Eigen::Vector3d frame_part(uint32_t idx, uint32_t id_part) const {
    CheckConsistentQuerry(idx, id_part);
    return id_part == 0 ? position(idx) : axis(idx, id_part - 1);
  }

  // Returns the body ID in the chain
  uint32_t rigid_body_id(std::string name) const {
    for (uint32_t i = 0; i < rigid_bodies_.size(); i++) {
      if (rigid_bodies_[i]->name() == name) {
        return i;
      }
    }
    throw std::runtime_error("KinematicChain : body name not found");
    return 0;
  }

  // Returns the number of active dofs in the chain
  uint32_t nb_active_dofs() const { return active_dofs_.size(); }

 protected:
  std::vector<std::shared_ptr<RigidBody>> rigid_bodies_;
  std::vector<std::shared_ptr<RigidBody>> active_dofs_;
  Eigen::Isometry3d base_;
};

}  // namespace bewego