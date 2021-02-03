// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <Eigen/Geometry>

Eigen::Matrix3d QuaternionToMatrix(const Eigen::VectorXd& q){
    assert(q.size() == 4);
    Eigen::Quaterniond quaternion;
    quaternion.x() = q[0];
    quaternion.y() = q[1];
    quaternion.z() = q[2];
    quaternion.w() = q[3];
    return quaternion.toRotationMatrix();
}

