// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com
#include <bewego/geometry.h>
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <random>

// using namespace bewego;
using std::cout;
using std::endl;

TEST(geometry, easy) {
  double roll = 1.5707, pitch = 0, yaw = 0.707;
  Eigen::Vector3d rpy;
  rpy.x() = roll;
  rpy.y() = pitch;
  rpy.z() = yaw;
  Eigen::Vector4d q = EulerToQuaternion(rpy);
  Eigen::Quaterniond quaternion;
  quaternion = q;

  Eigen::Vector3d euler = quaternion.toRotationMatrix().eulerAngles(2, 1, 0);
  Eigen::Vector3d euler2;
  euler2.x() = euler[2];
  euler2.y() = euler[1];
  euler2.z() = euler[0];
  ASSERT_LT((rpy - euler2).norm(), 1e-3);
  cout << " - rpy   : " << rpy.transpose() << endl;
  cout << " - euler : " << euler.transpose() << endl;
}

TEST(geometry, urdf) {
  // this is the convention found in URDF dom

  Eigen::Vector3d rpy = Eigen::Vector3d::Random();
  double roll = rpy[0];
  double pitch = rpy[1];
  double yaw = rpy[2];

  double phi = roll / 2.0;
  double the = pitch / 2.0;
  double psi = yaw / 2.0;

  Eigen::Vector4d q1;
  q1.x() = sin(phi) * cos(the) * cos(psi) - cos(phi) * sin(the) * sin(psi);
  q1.y() = cos(phi) * sin(the) * cos(psi) + sin(phi) * cos(the) * sin(psi);
  q1.z() = cos(phi) * cos(the) * sin(psi) - sin(phi) * sin(the) * cos(psi);
  q1.w() = cos(phi) * cos(the) * cos(psi) + sin(phi) * sin(the) * sin(psi);
  q1.normalize();

  Eigen::Vector4d q2 = EulerToQuaternion(rpy);
  ASSERT_LT((q1 - q2).norm(), 1e-3);

  rpy[0] = -1.57079632679;
  rpy[1] = -1.57079632679;
  rpy[2] = 0;

  cout << " - matrix   : " << endl
       << Eigen::Quaterniond(EulerToQuaternion(rpy)).toRotationMatrix() << endl;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}