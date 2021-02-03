// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <vector>
#include <iostream>

using std::cout;
using std::endl;

namespace bewego {

/*!\brief Represents bounds \in [lower, upper]. 
 * for a scalar variable
 */
class ScalarBound {
 public:
  ScalarBound() {}

  ScalarBound(double lower, double upper) :
    lower_(lower),
    upper_(upper)
    {}

  ScalarBound(const std::pair<double, double>& bounds) :
    lower_(bounds.first),
    upper_(bounds.second)
  {}

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

    enum JointType { ROTATIONAL, TRANSLATIONAL, FIXED } joint_type;

    RigidBody() {}
    RigidBody(
        const std::string& name,
        const std::string& joint_name,
        const Eigen::Affine3d& local_in_prev,
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


/*!\brief Represents a chain of rigid bodies
 */
class Robot {

public:
    Robot() {
        kinematic_chain_.clear();
    }

    void AddRigidBody(
        const std::string& name,
        const std::string& joint_name,
        const Eigen::Matrix4d& local_in_prev,
        const Eigen::Vector3d& joint_axis_in_local)
    {
        Eigen::Affine3d T;
        T.matrix() = local_in_prev;

        cout << "Add joint (" << name 
             << ") : " << endl << T.rotation() << endl;
        cout << "Add joint (" << name 
             << ") : " << endl << T.translation() << endl;
        kinematic_chain_.push_back(
            RigidBody(name, joint_name, T, joint_axis_in_local));
    }

    void SetAndUpdate(const Eigen::VectorXd& q) {
        SetConfiguration(q);
        ForwardKinematics();
    }

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

    Eigen::Vector3d get_position(uint32_t idx) const {
        return kinematic_chain_[idx].frame_in_base().translation();
    }

protected:

  std::vector<RigidBody> kinematic_chain_;
};

} // namespace bewego