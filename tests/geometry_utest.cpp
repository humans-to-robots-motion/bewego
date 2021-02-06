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
  Eigen::Vector3d euler = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);
  ASSERT_LT((rpy - euler).norm(), 1e-3);
  cout << " - rpy   : " << rpy.transpose() << endl;
  cout << " - euler : " << euler.transpose() << endl;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}