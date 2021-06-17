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
#include <bewego/workspace/collision_checking.h>

#include <memory>

namespace bewego {

/**
 * Interpolates the segments with keypoints given
 * a target total number
 */
std::vector<Eigen::VectorXd> GetKeyPoints(double nb_keypoints,
                                          const std::vector<Segment>& segments);

/**
 * Freeflying robot who's DoF is a single rigid body transformation
 * keypoints are used to define collision constraints.
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
  DifferentiableMapPtr keypoint_map(uint32_t i) const {
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
  // std::shared_ptr<const DifferentiableMap> ConstructCollisionChecker(
  //     const VectorOfMaps& surface_functions, double margin = 0);

 protected:
  virtual void CreateTaskMaps(
      const std::vector<Eigen::VectorXd>& keypoints) = 0;

  std::string name_;
  std::vector<Eigen::VectorXd> keypoints_;
  std::vector<double> radii_;
  std::vector<DifferentiableMapPtr> task_maps_;
};

/**
 * Freeflying robot who's DoF is a single rigid-body transformation
 * keypoints are used to define collision constraints.
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
 * Freeflying robot who's DoF is a single rigid-body transformation
 * keypoints are used to define collision constraints.
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

// Initializes a freeflyer based on keypoint structure.
std::shared_ptr<Freeflyer> CreateFreeFlyer(
    std::string name, const std::vector<Eigen::VectorXd>& keypoints,
    const std::vector<double>& radii);

std::shared_ptr<Freeflyer2D> MakeFreeflyer2D();
std::shared_ptr<Freeflyer3D> MakeFreeflyer3D();

/**
 * Gets the position from state
 */
class FreeFlyerTranslation : public DifferentiableMap {
 public:
  FreeFlyerTranslation(uint32_t n) : n_(n) {
    J_ = Eigen::MatrixXd::Zero(output_dimension(), input_dimension());
    J_.block(0, 0, n_, n_) = Eigen::MatrixXd::Identity(n_, n_);
    type_ = "FreeFlyerTranslation";
  }
  ~FreeFlyerTranslation() {}

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
    assert(q.size() == input_dimension());
    return q.segment(0, n_);
  }
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
    assert(q.size() == input_dimension());
    return J_;
  }
  // The input dimension is either 3 or 6 depending
  // on the dimension of the workspace
  virtual uint32_t input_dimension() const { return n_ + (n_ == 2 ? 1 : 3); }
  virtual uint32_t output_dimension() const { return n_; }

 protected:
  uint32_t n_;
};

}  // namespace bewego
