/*
 * Copyright (c) 2020
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
 *                                               Jim Mainprice Wed 4 Feb 2020
 */

#pragma once
#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/workspace/extent.h>

#include <Eigen/Geometry>

namespace bewego {

struct WorkspaceObject {
  virtual DifferentiableMapPtr ConstraintFunction() const = 0;
};

class Workspace {
 public:
  Workspace(const std::vector<std::shared_ptr<const WorkspaceObject>>& objects)
      : objects_(objects) {}

  VectorOfMaps ExtractSurfaceFunctions() const {
    VectorOfMaps signed_distance_functions;
    for (auto& obj : objects_) {
      signed_distance_functions.push_back(obj->ConstraintFunction());
    }
    return signed_distance_functions;
  }

  // Extract the signed distance field as the minimum of all
  // the SDF of the objects present in the workspace
  // This field has first order discontiuities but the values
  // should be continuous.
  DifferentiableMapPtr SignedDistanceField() const {
    return std::make_shared<Min>(ExtractSurfaceFunctions());
  }

 protected:
  std::vector<std::shared_ptr<const WorkspaceObject>> objects_;
};

// Evaluates the distance to an N-sphere
// WARNING: This function has been tested only for 2D and 3D cases.
// the hessian is still implemented with finite differences.
//
//        f(x) = | x - x_o | - r
//
class SphereDistance : public DifferentiableMap {
 public:
  // Constructor.
  SphereDistance(const Eigen::VectorXd& origin, double radius,
                 double dist_cutoff = std::numeric_limits<double>::max())
      : origin_(origin), radius_(radius), dist_(dist_cutoff) {}
  virtual ~SphereDistance() {}

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return Eigen::VectorXd::Constant(1, Evaluate(x));
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd J(1, input_dimension());
    Eigen::VectorXd g(input_dimension());
    Eigen::MatrixXd H(input_dimension(), input_dimension());
    Evaluate(x, &g);
    J.row(0) = g;
    return J;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    Eigen::VectorXd g(input_dimension());
    Eigen::MatrixXd H(input_dimension(), input_dimension());
    Evaluate(x, &g, &H);
    return H;
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
 * \brief Signed Distance Function (SDF) to a rectangular box
 * with orientation this implementation can give the true hessian
 * However when the closest point is a vertex of the box
 * the hessian can become infinite hence the dist_cutoff
 * which keeps the hessian bounded, when dist_cutoff is different
 * from double::max.
 */
class RectangleDistance : public DifferentiableMap {
 public:
  // Creates a distance function to the point x0. The dimensionality of x0
  // defines the dimensionality of this function.
  RectangleDistance(const Eigen::VectorXd& center,
                    const Eigen::VectorXd& dimension, double orientation,
                    double dist_cutoff = std::numeric_limits<double>::max())
      : RectangleDistance(center, dimension,
                          Eigen::Rotation2Dd(orientation).toRotationMatrix(),
                          dist_cutoff) {
    // CHECK_EQ(dim_, 2);
  }

  // Creates a distance function to the point x0. The dimensionality of x0
  // defines the dimensionality of this function.
  RectangleDistance(const Eigen::VectorXd& center,
                    const Eigen::VectorXd& dimension,
                    const Eigen::MatrixXd orientation,
                    double dist_cutoff = std::numeric_limits<double>::max()) {
    dim_ = uint32_t(center.size());
    center_ = center;
    dimensions_ = dimension;
    orientation_ = orientation;
    axis_ = Eigen::MatrixXd::Identity(dim_, dim_);
    dist_cutoff_ = dist_cutoff;
  }
  virtual ~RectangleDistance();

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

/**
 * \brief Represents basic circle workspace obstacle.
 */
class Circle : public WorkspaceObject {
 public:
  Circle(const Eigen::Vector2d& center, double radius)
      : center_(center), radius_(radius) {}
  Circle(const Circle& other)
      : center_(other.center_), radius_(other.radius_) {}
  Circle& operator=(const Circle& rhs) {
    center_ = rhs.center_;
    radius_ = rhs.radius_;
    return *this;
  }
  virtual ~Circle();
  DifferentiableMapPtr ConstraintFunction() const;
  const Eigen::Vector2d& center() const { return center_; }
  double radius() const { return radius_; }

 protected:
  Eigen::Vector2d center_;
  double radius_;
};

/**
 * \brief Represents basic box on the plane
 */
class Rectangle : public WorkspaceObject {
 public:
  Rectangle(const Eigen::Vector2d& center, const Eigen::Vector2d& dimensions,
            double orientation) {
    center_ = center;
    dimensions_ = dimensions;
    orientation_ = orientation;
  }
  Rectangle(const Eigen::Vector2d& top_left,
            const Eigen::Vector2d& bottom_right);
  Rectangle(const Rectangle& other)
      : center_(other.center_),
        dimensions_(other.dimensions_),
        orientation_(other.orientation_) {}
  Rectangle& operator=(const Rectangle& rhs) {
    center_ = rhs.center_;
    dimensions_ = rhs.dimensions_;
    orientation_ = rhs.orientation_;
    return *this;
  }
  virtual ~Rectangle();
  DifferentiableMapPtr ConstraintFunction() const;
  const Eigen::Vector2d& center() const { return center_; }
  const Eigen::Vector2d& dimensions() const { return dimensions_; }
  double orientation() const { return orientation_; }

  extent_t extent() const {
    double half_dim_x = dimensions_.x() / 2.;
    double half_dim_y = dimensions_.y() / 2.;
    return extent_t(center_.x() - half_dim_x, center_.x() + half_dim_x,
                    center_.y() - half_dim_y, center_.y() + half_dim_y);
  }

 protected:
  Eigen::Vector2d center_;
  Eigen::Vector2d dimensions_;
  double orientation_;
};

}  // namespace bewego
