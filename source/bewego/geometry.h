// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <Eigen/Geometry>

Eigen::Matrix3d QuaternionToMatrix(const Eigen::VectorXd& q) {
  assert(q.size() == 4);
  Eigen::Quaterniond quaternion;
  quaternion.x() = q[0];
  quaternion.y() = q[1];
  quaternion.z() = q[2];
  quaternion.w() = q[3];
  return quaternion.toRotationMatrix();
}

Eigen::Vector4d EulerToQuaternion(const Eigen::VectorXd& rpy) {
  assert(rpy.size() == 3);
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) * 
      Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ());
  Eigen::Vector4d quaternion;
  quaternion[0] = q.x();
  quaternion[1] = q.y();
  quaternion[2] = q.z();
  quaternion[3] = q.w();
  return quaternion;
}
