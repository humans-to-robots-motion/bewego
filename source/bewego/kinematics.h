// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <vector>
#include <iostream>

namespace bewego {

/*!\brief Represents bounds \in [lower, upper]. 
 * for a scalar variable
 */
class ScalarBound {
 public:
  ScalarBound();

  ScalarBound(double lower, double upper);
  ScalarBound(const std::pair<double, double>& bounds);

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

class RigidBody {

public:
    enum JointType { ROTATIONAL, TRANSLATIONAL, FIXED } joint_type;

  RigidBody() {}


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
        std::cerr << "This joint type is not supported: "
                  << joint_type;
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
  const Eigen::Affine3d& frame_in_base() const { return frame_in_base_; }

protected:

  std::string name_;  // Rigid body name in URDF.
  ScalarBound joint_bounds_;  // Parent joint limits
  std::string joint_name_; // Joint body name

  Eigen::Affine3d frame_in_base_;      // FK solution
  Eigen::Affine3d frame_in_local_;     // Joint transformed frame
  Eigen::Affine3d local_in_prev_;      // Prev to curr transform

  // Info about the joint axis. 
  // If it's axis aligned, then we can skip some computation.
  Eigen::Vector3d joint_axis_in_local_;
  Eigen::Vector3d joint_axis_in_base_;
};


class Robot {

public:
    Robot() {}

    void SetConfiguration(const Eigen::VectorXd& q) {
        for(uint32_t i=0; i < kinematic_chain_.size(); i++) {
            kinematic_chain_[i].SetDoF(q[i]);
        }
    }
    void ForwardKinematics() {
        auto& parent = kinematic_chain_[0];
        for(uint32_t i= 1; i<kinematic_chain_.size(); i++) {
            auto& rigid_body = kinematic_chain_[i];
            rigid_body.Propagate(parent.frame_in_base());
            parent = rigid_body;
        }
    }
    void Jacobian() {}

protected:

  std::vector<RigidBody> kinematic_chain_;
};

} // namespace bewego