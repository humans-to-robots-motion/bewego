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
 *                                                              Thu 15 Apr 2021
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/motion/differentiable_kinematics.h>

namespace bewego {
//-----------------------------------------------------------------------------
// Position implementation.
//-----------------------------------------------------------------------------

Position::~Position() {}
Eigen::VectorXd Position::Forward(const Eigen::VectorXd& x) const {
  assert(x.size() == input_dimension());
  return x.segment(0, n_);
}
Eigen::MatrixXd Position::Jacobian(const Eigen::VectorXd& x) const {
  assert(x.size() == input_dimension());
  return J_;
}

//-----------------------------------------------------------------------------
// Velocity implementation.
//-----------------------------------------------------------------------------

Velocity::~Velocity() {}
Eigen::VectorXd Velocity::Forward(const Eigen::VectorXd& x) const {
  assert(x.size() == input_dimension());
  return x.segment(n_, n_);
}
Eigen::MatrixXd Velocity::Jacobian(const Eigen::VectorXd& x) const {
  assert(x.size() == input_dimension());
  return J_;
}

//-----------------------------------------------------------------------------
// VelocityInFrame implementation.
//-----------------------------------------------------------------------------

Eigen::VectorXd VelocityInFrame::Forward(const Eigen::VectorXd& q) const {
  assert(q.size() == input_dimension());
  Eigen::Isometry3d T;
  T.translation() = q.segment(0, 3);       // position
  T.linear().col(0) = q.segment(3, 3);     // x_axis
  T.linear().col(1) = q.segment(6, 3);     // y_axis
  T.linear().col(2) = q.segment(9, 3);     // z_axis
  Eigen::Vector3d vel = q.segment(12, 3);  // d/dt pos
  return T.linear().transpose() * vel;
}

Eigen::MatrixXd VelocityInFrame::Jacobian(const Eigen::VectorXd& q) const {
  assert(q.size() == input_dimension());
  J_.block<1, 3>(0, 12) = q.segment(3, 3);  // x_axis
  J_.block<1, 3>(1, 12) = q.segment(6, 3);  // y_axis
  J_.block<1, 3>(2, 12) = q.segment(9, 3);  // z_axis
  Eigen::Vector3d vel = q.segment(12, 3);   // d/dt pos
  J_.block<1, 3>(0, 3) = vel;
  J_.block<1, 3>(1, 6) = vel;
  J_.block<1, 3>(2, 9) = vel;
  return J_;
}

//-----------------------------------------------------------------------------
// HomogeneousTransform2d implementation.
//-----------------------------------------------------------------------------

Eigen::VectorXd HomogeneousTransform2d::Forward(
    const Eigen::VectorXd& q) const {
  assert(q.size() == input_dimension());
  Eigen::Isometry2d T;
  T.translation() = q.segment(0, 2);  // position
  T.linear() = Eigen::Rotation2Dd(q[2]).toRotationMatrix();
  return T * p0_;
}

Eigen::MatrixXd HomogeneousTransform2d::Jacobian(
    const Eigen::VectorXd& q) const {
  assert(q.size() == input_dimension());
  double c = std::cos(q[2]);
  double s = std::sin(q[2]);
  J_(0, 0) = 1;
  J_(1, 1) = 1;
  J_(0, 2) = -s * p0_.x() - c * p0_.y();
  J_(1, 2) = c * p0_.x() - s * p0_.y();
  return J_;
}

//-----------------------------------------------------------------------------
// HomogeneousTransform2d implementation.
//-----------------------------------------------------------------------------

// Returns Transform
// We use the Euler angel convention 3-2-1, which is found
// TODO it would be nice to match the ROS convention
// we simply use this one because it was available as derivation
// in termes for sin and cos. It seems that ROS dos
// Static-Z-Y-X so it should be the same. Still needs to test.
Eigen::Isometry3d HomogeneousTransform3d::Transform(
    const Eigen::VectorXd& q) const {
  assert(q.size() == input_dimension());
  Eigen::Isometry3d T;
  T.translation() = q.segment(0, 3);  // translation
  Eigen::Matrix3d R;                  // rotation
  // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  // See wikipedia...
  //  R = Eigen::AngleAxisd(q[3], Eigen::Vector3d::UnitZ()) *
  //      Eigen::AngleAxisd(q[4], Eigen::Vector3d::UnitY()) *
  //      Eigen::AngleAxisd(q[5], Eigen::Vector3d::UnitX());

  double cx = std::cos(q[3]);  // cos psi
  double sx = std::sin(q[3]);  // sin psi
  double ct = std::cos(q[4]);  // cos theta
  double st = std::sin(q[4]);  // sin theta
  double cp = std::cos(q[5]);  // cos phi
  double sp = std::sin(q[5]);  // sin phi
  // TODO
  R.row(0) << ct * cx, -cp * sx + sp * st * cx, sp * sx + cp * st * cx;
  R.row(1) << ct * sx, cp * cx + sp * st * sx, -sp * cx + cp * st * sx;
  R.row(2) << -st, sp * ct, cp * ct;
  T.linear() = R;
  return T;
}

Eigen::VectorXd HomogeneousTransform3d::Forward(
    const Eigen::VectorXd& q) const {
  return Transform(q) * p0_;
}

Eigen::MatrixXd HomogeneousTransform3d::Jacobian(
    const Eigen::VectorXd& q) const {
  assert(q.size() == input_dimension());
  Eigen::Vector3d p1, p2, p3, p4, p5, p6, p8, p9;

  double cx = std::cos(q[3]);  // cos psi
  double sx = std::sin(q[3]);  // sin psi
  double ct = std::cos(q[4]);  // cos theta
  double st = std::sin(q[4]);  // sin theta
  double cp = std::cos(q[5]);  // cos phi
  double sp = std::sin(q[5]);  // sin phi

  J_(0, 0) = 1;  // dx/dx
  J_(1, 1) = 1;  // dy/dy
  J_(2, 2) = 1;  // dz/dz

  p1 << ct * -sx, -cp * cx - sp * st * sx, sp * cx - cp * st * sx;  // dx/dpsi
  p2 << -st * cx, sp * ct * cx, cp * ct * cx;                       // dx/dtheta
  p3 << 0, sp * sx + cp * st * cx, cp * sx + -sp * st * cx;         // dx/dphi
  p4 << ct * cx, -cp * sx + sp * st * cx, sp * sx + cp * st * cx;   // dy/dpsi
  p5 << -st * sx, sp * ct * sx, cp * ct * sx;                       // dy/dtheta
  p6 << 0, -sp * cx + cp * st * sx, -cp * cx + -sp * st * sx;       // dy/dphi
  p8 << -ct, -sp * st, -cp * st;                                    // dz/dtheta
  p9 << 0, cp * ct, -sp * ct;                                       // dz/dphi

  J_(0, 3) = p1.transpose() * p0_;
  J_(0, 4) = p2.transpose() * p0_;
  J_(0, 5) = p3.transpose() * p0_;
  J_(1, 3) = p4.transpose() * p0_;
  J_(1, 4) = p5.transpose() * p0_;
  J_(1, 5) = p6.transpose() * p0_;
  J_(2, 3) = 0;
  J_(2, 4) = p8.transpose() * p0_;
  J_(2, 5) = p9.transpose() * p0_;

  return J_;
}

}  // namespace bewego