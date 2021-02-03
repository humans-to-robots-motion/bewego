// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/kinematics.h>

namespace bewego {

RigidBody::RigidBody(
    const std::string& name,
    const std::string& joint_name,
    const Eigen::Affine3d& local_in_prev,
    const Eigen::Vector3d& joint_axis_in_local) :
        name_(name),
        joint_bounds_(0, 0),
        joint_name_(joint_name),
        joint_type(ROTATIONAL),
        frame_in_base_(Eigen::Affine3d::Identity()),
        frame_in_local_(Eigen::Affine3d::Identity()),
        local_in_prev_(local_in_prev),
        joint_axis_in_local_(joint_axis_in_local),
        joint_axis_in_base_(Eigen::Vector3d::Zero()) {}

}