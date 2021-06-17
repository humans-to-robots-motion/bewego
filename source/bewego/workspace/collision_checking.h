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
 *                                                              Thu 17 Jun 2021
 */
// author: Jim Mainprice, mainprice@gmail.com

#pragma once

#include <bewego/derivatives/differentiable_map.h>

#include <memory>

namespace bewego {

struct Segment {
  Eigen::VectorXd x1;
  Eigen::VectorXd x2;
  double length() const { return (x1 - x2).norm(); }
  Eigen::VectorXd interpolate(double alpha) const {
    return alpha * x1 + (1 - alpha) * x2;
  }
};

struct CollisionPoint {
  CollisionPoint() {}
  CollisionPoint(const CollisionPoint& point)
      : task_map(point.task_map), radius(point.radius) {}
  CollisionPoint(DifferentiableMapPtr m, double r) : task_map(m), radius(r) {}
  DifferentiableMapPtr task_map;
  double radius;
};

using VectorOfCollisionPoints = std::vector<CollisionPoint>;

// Creates a collision checker for a robot and a workspace
DifferentiableMapPtr ConstructCollisionChecker(
    const VectorOfCollisionPoints& collision_points,
    const VectorOfMaps& surface_functions, double margin);

/**
 * Collision constraint function that averages all collision points.
 * Also provides an interface for dissociating each constraint
 * The vector of collision points contains a pointer to each
 * foward kinematics map (task map).
 * to get the surfaces simply use workspace->ExtractSurfaceFunctions
 * there is a surface per object in the workspace
 */
class SmoothCollisionConstraint : public CachedDifferentiableMap {
 public:
  SmoothCollisionConstraint(const VectorOfMaps& surfaces, double gamma,
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

/**
 * Collision constraint function that averages all collision points.
 * Also provides an interface for dissociating each constraint
 * The vector of collision points contains a pointer to each
 * foward kinematics map (task map).
 * to get the surfaces simply use workspace->ExtractSurfaceFunctions
 * there is a surface per object in the workspace
 */
class SmoothCollisionPointsConstraint {
 public:
  SmoothCollisionPointsConstraint(
      const VectorOfCollisionPoints& collision_points,
      const VectorOfMaps& surfaces, double gamma = 100.);
  virtual ~SmoothCollisionPointsConstraint();

  /**
   * @brief constraints
   * @return a vector of signed distance function defined over
   * configuration space. One for each collsion point
   */
  const VectorOfMaps& constraints() const { return signed_distance_functions_; }

  /**
   * @brief smooth_constraint
   * @return a softmin function of the constraint
   */
  DifferentiableMapPtr smooth_constraint() const { return f_; }

 protected:
  VectorOfCollisionPoints collision_points_;
  VectorOfMaps surfaces_;
  double margin_;
  VectorOfMaps signed_distance_functions_;
  double gamma_;
  DifferentiableMapPtr f_;
  uint32_t n_;
};
}  // namespace bewego
