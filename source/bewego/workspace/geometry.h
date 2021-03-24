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
  // This is supposed to be the URDF convention
  // tested against pybullet. If it's correct there it
  // it should be ok.
  assert(rpy.size() == 3);
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX());
  Eigen::Vector4d quaternion;
  quaternion[0] = q.x();
  quaternion[1] = q.y();
  quaternion[2] = q.z();
  quaternion[3] = q.w();
  return quaternion;
}
