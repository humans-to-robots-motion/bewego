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

#include <bewego/derivatives/combination_operators.h>
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/motion/differentiable_kinematics.h>
#include <bewego/motion/freeflyers.h>

using std::cout;
using std::endl;

namespace bewego {

//-----------------------------------------------------------------------------
// Freeflyer function implementation.
//-----------------------------------------------------------------------------

Freeflyer::~Freeflyer() {}

// Get Collision Points
VectorOfCollisionPoints Freeflyer::GetCollisionPoints() const {
  VectorOfCollisionPoints collision_points;
  for (uint32_t i = 0; i < task_maps_.size(); i++) {
    CollisionPoint point(task_maps_[i], radii_[i]);
    collision_points.push_back(point);
  }
  return collision_points;
}

// Creates a collision checker for a robot and a workspace
DifferentiableMapPtr Freeflyer::ConstructCollisionChecker(
    const VectorOfMaps& surface_functions, double margin) {
  const VectorOfCollisionPoints& collision_points = GetCollisionPoints();
  VectorOfMaps signed_distance_functions;
  for (auto& surface : surface_functions) {
    for (auto& sphere : collision_points) {
      double offset = sphere.radius + margin;
      signed_distance_functions.push_back(
          ComposedWith(sphere.task_map, surface - offset));
    }
  }
  return std::make_shared<Min>(signed_distance_functions);
}

//-----------------------------------------------------------------------------
// Freeflyer2D function implementation.
//-----------------------------------------------------------------------------

Freeflyer2D::~Freeflyer2D() {}
void Freeflyer2D::CreateTaskMaps(
    const std::vector<Eigen::VectorXd>& keypoints) {
  task_maps_.resize(keypoints.size());
  for (uint32_t i = 0; i < keypoints.size(); i++) {
    cout << " - keypoint[" << i << "] : " << keypoints[i].transpose() << endl;
    task_maps_[i] = std::make_shared<HomogeneousTransform2d>(keypoints[i]);
  }
}

//-----------------------------------------------------------------------------
// Freeflyer3D function implementation.
//-----------------------------------------------------------------------------

Freeflyer3D::~Freeflyer3D() {}
void Freeflyer3D::CreateTaskMaps(
    const std::vector<Eigen::VectorXd>& keypoints) {
  task_maps_.resize(keypoints.size());
  for (uint32_t i = 0; i < keypoints.size(); i++) {
    cout << " - keypoint[" << i << "] : " << keypoints[i].transpose() << endl;
    task_maps_[i] = std::make_shared<HomogeneousTransform3d>(keypoints[i]);
  }
}

//-----------------------------------------------------------------------------
// FreeFlyerCollisionConstraints function implementation.
//-----------------------------------------------------------------------------

FreeFlyerCollisionConstraints::FreeFlyerCollisionConstraints(
    std::shared_ptr<const Freeflyer> freeflyer, const VectorOfMaps& surfaces,
    double gamma)
    : collision_points_(freeflyer->GetCollisionPoints()),
      surfaces_(surfaces),
      margin_(0.),
      gamma_(gamma) {
  // Check input dimensions
  assert(collision_points_.size() > 0);
  auto task_map = collision_points_.back().task_map;
  assert(task_map.get() != nullptr);
  n_ = task_map->input_dimension();
  for (auto& sphere : collision_points_) {
    assert(n_ == sphere.task_map->input_dimension());
  }

  // First interate through all the surfaces in the environment
  // Add a constraint per keypoint on the freeflyer.
  // TODO have a different model for the robot (with capsules or ellipsoids)
  VectorOfMaps maps;
  for (auto& surface : surfaces_) {
    for (auto& sphere : collision_points_) {
      auto task_space =
          ComposedWith(sphere.task_map, surface - (sphere.radius + margin_));
      signed_distance_functions_.push_back(task_space);
      maps.push_back(task_space);
    }
  }
  auto smooth_min = std::make_shared<NegLogSumExp>(maps.size(), gamma_);
  auto stack = std::make_shared<CombinedOutputMap>(maps);
  f_ = ComposedWith(stack, smooth_min);
}

// Destructor.
FreeFlyerCollisionConstraints::~FreeFlyerCollisionConstraints() {}

std::vector<Eigen::VectorXd> GetKeyPoints(
    double nb_keypoints, const std::vector<Segment>& segments) {
  double length = 0.;
  for (auto s : segments) {
    length += s.length();
  }
  std::vector<Eigen::VectorXd> keypoints;
  double dl = length / nb_keypoints;
  for (auto s : segments) {
    double k = std::floor(s.length() / dl);
    double d_alpha = (1. / k);
    double d = 0.;
    for (uint32_t i = 0; i < k; i++) {
      keypoints.push_back(s.interpolate(d));
      d += d_alpha;
    }
  }
  return keypoints;
}

}  // namespace bewego
