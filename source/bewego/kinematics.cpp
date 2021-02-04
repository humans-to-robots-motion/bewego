// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/kinematics.h>

using std::cout;
using std::endl;

namespace bewego {

RigidBody::RigidBody(const std::string& name, const std::string& joint_name,
                     const Eigen::Affine3d& local_in_prev,
                     const Eigen::Vector3d& joint_axis_in_local)
    : debug_(false),
      name_(name),
      joint_bounds_(0, 0),
      joint_name_(joint_name),
      joint_type(ROTATIONAL),
      frame_in_base_(Eigen::Affine3d::Identity()),
      frame_in_local_(Eigen::Affine3d::Identity()),
      local_in_prev_(local_in_prev),
      joint_axis_in_local_(joint_axis_in_local),
      joint_axis_in_base_(Eigen::Vector3d::Zero()) {
  if (debug_) {
    cout << "***** Create Rigid body ** " << endl;
    cout << " -- local_in_prev_  : " << endl << local_in_prev_.matrix() << endl;
    cout << " -- joint_axis_in_local_  : " << joint_axis_in_local_.transpose()
         << endl;
  }
}

Eigen::MatrixXd Robot::JacobianPosition(int link_index) const {
  Eigen::MatrixXd J(3, kinematic_chain_.size());
  auto x = kinematic_chain_[link_index].joint_origin_in_base();
  for (int j = J.cols() - 1; j >= 0; j--) {
    auto joint_origin = kinematic_chain_[j].joint_origin_in_base();
    auto& joint_axis = kinematic_chain_[j].joint_axis_in_base();
    J.col(j) = joint_axis.cross(x - joint_origin);
  }
  return J;
}

}  // namespace bewego