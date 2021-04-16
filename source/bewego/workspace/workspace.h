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
#include <bewego/derivatives/combination_operators.h>
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/workspace/extent.h>
#include <bewego/workspace/geometry.h>

#include <Eigen/Geometry>

namespace bewego {

struct WorkspaceObject {
  virtual DifferentiableMapPtr ConstraintFunction() const = 0;
};

using WorkspaceObjectPtr = std::shared_ptr<const WorkspaceObject>;
using VectorOfWorkpaceObjects = std::vector<WorkspaceObjectPtr>;

class Workspace {
 public:
  Workspace(const VectorOfWorkpaceObjects& objects)
      : objects_(objects), dimension_(2) {}

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

  // Ambient space dimension of the workspace (2D or 3D)
  // For now we have only 2 dimensional cases
  // but it should be easy to extend the rectangles and circles to the 3D case
  uint32_t dimension() const { return dimension_; }

  // Evaluate weather a given point is inside the sphere worlds
  bool InCollision(const Eigen::VectorXd& p) const {
    for (const auto& object : objects_) {
      if (object->ConstraintFunction()->ForwardFunc(p) < 0) {
        return true;
      }
    }
    return false;
  }

 protected:
  uint32_t dimension_;
  std::vector<std::shared_ptr<const WorkspaceObject>> objects_;
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

/**
 * \brief  Represents sphere in Cartesian space
 */
class Sphere : public WorkspaceObject {
 public:
  Sphere(const Eigen::Vector3d& center, double radius)
      : center_(center), radius_(radius) {}
  Sphere(const Sphere& other)
      : center_(other.center_), radius_(other.radius_) {}

  DifferentiableMapPtr ConstraintFunction() const {
    return std::make_shared<SphereDistance>(center_, radius_);
  }

  const Eigen::Vector3d& center() const { return center_; }
  double radius() const { return radius_; }

 protected:
  Eigen::Vector3d center_;
  double radius_;
};

/**
 Collision constraint function that averages all collision points.
 Also provides an interface for dissociating each constraint
 The vector of collision points contains a pointer to each
 foward kinematics map (task map).
 to get the surfaces simply use workspace->ExtractSurfaceFunctions
 there is a surface per object in the workspace
 */
class SmoothCollisionConstraints : public CachedDifferentiableMap {
 public:
  SmoothCollisionConstraints(const VectorOfMaps& surfaces, double gamma,
                             double margin = 0);

  /**
   * @brief constraints
   * @return a vector of signed distance function defined over
   * configuration space. One for each collsion point
   */

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return f_->input_dimension(); }

  Eigen::VectorXd Forward_(const Eigen::VectorXd& x) const {
    return f_->Forward(x);
  }
  Eigen::MatrixXd Jacobian_(const Eigen::VectorXd& x) const {
    return f_->Jacobian(x);
  }
  Eigen::MatrixXd Hessian_(const Eigen::VectorXd& x) const {
    return f_->Hessian(x);
  }

  virtual VectorOfMaps nested_operators() const { return VectorOfMaps({f_}); }

  VectorOfMaps constraints() const { return signed_distance_functions_; }

 protected:
  DifferentiableMapPtr ConstructSmoothConstraint();
  double margin_;
  VectorOfMaps signed_distance_functions_;
  double gamma_;
  DifferentiableMapPtr f_;
};

}  // namespace bewego
