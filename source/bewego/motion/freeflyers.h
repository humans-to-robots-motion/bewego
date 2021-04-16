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

#pragma once

#include <bewego/derivatives/differentiable_map.h>

#include <memory>

namespace bewego {

using TaskMap = std::shared_ptr<const DifferentiableMap>;

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
  CollisionPoint(TaskMap m, double r) : task_map(m), radius(r) {}
  TaskMap task_map;
  double radius;
};

using VectorOfCollisionPoints = std::vector<CollisionPoint>;

/**
   Interpolates the segments with keypoints given
   a target total number
 */
std::vector<Eigen::VectorXd> GetKeyPoints(double nb_keypoints,
                                          const std::vector<Segment>& segments);

/**
   Freeflying robot who's DoF is a single rigid body transformation
   keypoints are used to define collision constraints.
 */
class Freeflyer {
 public:
  Freeflyer(std::string name, const std::vector<Eigen::VectorXd>& keypoints,
            const std::vector<double>& radii)
      : name_(name), keypoints_(keypoints), radii_(radii) {}
  Freeflyer(const Freeflyer& other)
      : name_(other.name_),
        keypoints_(other.keypoints_),
        radii_(other.radii_) {}
  virtual ~Freeflyer();

  // Returns the keypoints used to describe the geometry
  const std::vector<Eigen::VectorXd>& keypoints() const { return keypoints_; }

  // Returns the kinematics map to a point on the structure
  TaskMap keypoint_map(uint32_t i) const {
    assert(i < task_maps_.size());
    return task_maps_[i];
  }

  // Returns keypoints radii
  double keypoint_radius(uint32_t i) const {
    assert(i < radii_.size());
    return radii_[i];
  }

  // Returns the name of the robot
  const std::string& name() const { return name_; }

  // Returns dimension of workspace
  virtual uint32_t n() const = 0;

  // Clone Robot
  virtual std::shared_ptr<Freeflyer> Clone() const = 0;

  // Get Collision Points
  virtual VectorOfCollisionPoints GetCollisionPoints() const;

  // Get a Collision checker For that FreeFlyer
  std::shared_ptr<const DifferentiableMap> ConstructCollisionChecker(
      const VectorOfMaps& surface_functions, double margin = 0);

 protected:
  virtual void CreateTaskMaps(
      const std::vector<Eigen::VectorXd>& keypoints) = 0;

  std::string name_;
  std::vector<Eigen::VectorXd> keypoints_;
  std::vector<double> radii_;
  std::vector<TaskMap> task_maps_;
};

/**
   Freeflying robot who's DoF is a single rigid-body transformation
   keypoints are used to define collision constraints.
 */
class Freeflyer2D : public Freeflyer {
 public:
  Freeflyer2D(std::string name, const std::vector<Eigen::VectorXd>& keypoints,
              const std::vector<double>& radii)
      : Freeflyer(name, keypoints, radii) {
    CreateTaskMaps(keypoints);
  }
  Freeflyer2D(const Freeflyer2D& other) : Freeflyer(other) {
    CreateTaskMaps(other.keypoints());
  }
  virtual ~Freeflyer2D();
  virtual void CreateTaskMaps(const std::vector<Eigen::VectorXd>& keypoints);
  virtual uint32_t n() const { return 2; }
  virtual std::shared_ptr<Freeflyer> Clone() const {
    return std::make_shared<Freeflyer2D>(*this);
  }
};

/**
   Freeflying robot who's DoF is a single rigid-body transformation
   keypoints are used to define collision constraints.
 */
class Freeflyer3D : public Freeflyer {
 public:
  Freeflyer3D(std::string name, const std::vector<Eigen::VectorXd>& keypoints,
              const std::vector<double>& radii)
      : Freeflyer(name, keypoints, radii) {
    CreateTaskMaps(keypoints);
  }
  Freeflyer3D(const Freeflyer3D& other) : Freeflyer(other) {
    CreateTaskMaps(other.keypoints());
  }
  virtual ~Freeflyer3D();
  virtual void CreateTaskMaps(const std::vector<Eigen::VectorXd>& keypoints);
  virtual uint32_t n() const { return 3; }
  virtual std::shared_ptr<Freeflyer> Clone() const {
    return std::make_shared<Freeflyer3D>(*this);
  }
};

// Collision constraint function that averages all collision points.
// Also provides an interface for dissociating each constraint
// The vector of collision points contains a pointer to each
// foward kinematics map (task map).
// to get the surfaces simply use workspace->ExtractSurfaceFunctions
// there is a surface per object in the workspace
class FreeFlyerCollisionConstraints {
 public:
  FreeFlyerCollisionConstraints(std::shared_ptr<const Freeflyer> freeflyer,
                                const VectorOfMaps& surfaces,
                                double gamma = 100.);
  virtual ~FreeFlyerCollisionConstraints();

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
