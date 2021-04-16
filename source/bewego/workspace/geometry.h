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

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/combination_operators.h>
#include <bewego/derivatives/differentiable_map.h>

#include <Eigen/Geometry>

namespace bewego {

inline Eigen::Matrix3d QuaternionToMatrix(const Eigen::VectorXd& q) {
  assert(q.size() == 4);
  Eigen::Quaterniond quaternion;
  quaternion.x() = q[0];
  quaternion.y() = q[1];
  quaternion.z() = q[2];
  quaternion.w() = q[3];
  return quaternion.toRotationMatrix();
}

/**
 *   This is supposed to be the URDF convention
 *   tested against pybullet. If it's correct there it
 *   it should be ok.
 */
inline Eigen::Vector4d EulerToQuaternion(const Eigen::VectorXd& rpy) {
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

/**
 *    Evaluates the distance to an N-sphere
 *
 *        f(x) = | x - x_o | - r
 *
 * WARNING: This function has been tested only for 2D and 3D cases.
 * the hessian is still implemented with finite differences.
 */
class SphereDistance : public DifferentiableMap {
 public:
  // Constructor.
  SphereDistance(const Eigen::VectorXd& origin, double radius,
                 double dist_cutoff = std::numeric_limits<double>::max())
      : origin_(origin), radius_(radius), dist_(dist_cutoff) {
    PreAllocate();
    type_ = "SphereDistance";
  }
  virtual ~SphereDistance() {}

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    y_[0] = Evaluate(x);
    return y_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Evaluate(x, &g_);
    return g_.transpose();
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    Evaluate(x, &g_, &H_);
    return H_;
  }

  virtual double Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g,
                          Eigen::MatrixXd* H) const;
  virtual double Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g) const;
  virtual double Evaluate(const Eigen::VectorXd& x) const;

  // custom accessors.
  const Eigen::VectorXd& origin() const { return origin_; }
  double radius() const { return radius_; }

  // This class is meant for n spheres.
  // WARNING: Only tested for 2D and 3D
  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return origin_.size(); }

  void set_alpha(double v) { dist_ = v; }

 protected:
  Eigen::VectorXd origin_;
  double radius_;
  double dist_;
};

/**
 *  Signed Distance Function (SDF) to a box (or rectangle in 2D)
 *  with orientation this implementation can give the true hessian
 *
 *  However when the closest point is a vertex of the box
 *  the hessian can become infinite hence the dist_cutoff
 *  which keeps the hessian bounded, when dist_cutoff is different
 *  from double::max.
 */
class BoxDistance : public DifferentiableMap {
 public:
  // Creates a distance function to the point x0. The dimensionality of x0
  // defines the dimensionality of this function.
  BoxDistance(const Eigen::VectorXd& center, const Eigen::VectorXd& dimension,
              double orientation,
              double dist_cutoff = std::numeric_limits<double>::max())
      : BoxDistance(center, dimension,
                    Eigen::Rotation2Dd(orientation).toRotationMatrix(),
                    dist_cutoff) {
    // CHECK_EQ(dim_, 2);
    type_ = "BoxDistance";
  }

  // Creates a distance function to the point x0. The dimensionality of x0
  // defines the dimensionality of this function.
  BoxDistance(const Eigen::VectorXd& center, const Eigen::VectorXd& dimension,
              const Eigen::MatrixXd orientation,
              double dist_cutoff = std::numeric_limits<double>::max()) {
    dim_ = uint32_t(center.size());
    center_ = center;
    dimensions_ = dimension;
    orientation_ = orientation;
    axis_ = Eigen::MatrixXd::Identity(dim_, dim_);
    dist_cutoff_ = dist_cutoff;
    type_ = "BoxDistance";
  }
  virtual ~BoxDistance();

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return Eigen::VectorXd::Constant(1, Evaluate(x));
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd J(1, input_dimension());
    Eigen::VectorXd g(input_dimension());
    Eigen::MatrixXd H(input_dimension(), input_dimension());
    Evaluate(x, &g, &H);
    J.row(0) = g;
    return J;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    Eigen::VectorXd g(input_dimension());
    Eigen::MatrixXd H(input_dimension(), input_dimension());
    Evaluate(x, &g, &H);
    return H;
  }

  virtual double Evaluate(const Eigen::VectorXd& x) const;
  virtual double Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g,
                          Eigen::MatrixXd* H) const;

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return dim_; }

 protected:
  uint32_t dim_;
  Eigen::VectorXd center_;
  Eigen::VectorXd dimensions_;
  Eigen::MatrixXd orientation_;
  Eigen::MatrixXd axis_;
  // for the hessian of the sphere distance otherwise
  // we have a theoretical inifinte curvature
  double dist_cutoff_;
};

}  // namespace bewego
