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
// CreateFreeFlyer
//-----------------------------------------------------------------------------

std::shared_ptr<Freeflyer> CreateFreeFlyer(
    std::string name, const std::vector<Eigen::VectorXd>& keypoints,
    const std::vector<double>& radii) {
  if (keypoints.size() != radii.size()) {
    throw std::runtime_error("Freeflyer : keypoints size != radii size");
  }
  uint32_t dim = keypoints.back().size();
  if (dim != 2 && dim != 3) {
    throw std::runtime_error(
        "Freeflyer : dimension missmatch (should be 2 or 3) got : " +
        std::to_string(dim));
  }
  for (uint32_t i = 0; i < keypoints.size(); i++) {
    if (keypoints[i].size() != dim) {
      throw std::runtime_error(
          "Freeflyer : dimension keypoint missmatch (should be 2 or 3) got :" +
          std::to_string(keypoints[i].size()));
    }
  }
  if (dim == 2) {
    return std::make_shared<Freeflyer2D>(name, keypoints, radii);
  } else {
    return std::make_shared<Freeflyer3D>(name, keypoints, radii);
  }
}

//-----------------------------------------------------------------------------
// Freeflyer test
//-----------------------------------------------------------------------------

std::shared_ptr<Freeflyer2D> MakeFreeflyer2D() {
  // L shaped planar freeflyer, hard-coded for testing
  // "s0" : [ 5.5, 0.5, 0.5, 0.5 ]
  // "s1" : [ 0.5, 0.5, 0.5, 5.5 ]
  std::vector<Segment> segments(2);
  segments[0].x1 = Eigen::Vector2d(5.5, 0.5);
  segments[0].x2 = Eigen::Vector2d(0.5, 0.5);
  segments[1].x1 = Eigen::Vector2d(0.5, 0.5);
  segments[1].x2 = Eigen::Vector2d(0.5, 5.5);
  uint32_t nb_keypoints = 10;
  double radii = 0.7;
  double scale = 0.03;

  return std::make_shared<Freeflyer2D>(
      "freeflyer_2", GetKeyPoints(nb_keypoints, segments),
      std::vector<double>(nb_keypoints, scale * radii));
}

std::shared_ptr<Freeflyer3D> MakeFreeflyer3D() {
  // L shaped planar freeflyer, hard-coded for testing
  // "s0": [5.5, 0.5, 0, 0.5, 0.5, 0]
  // "s1": [0.5, 0.5, 0, 0.5, 5.5, 0]
  std::vector<Segment> segments(2);
  segments[0].x1 = Eigen::Vector3d(5.5, 0.5, 0);
  segments[0].x2 = Eigen::Vector3d(0.5, 0.5, 0);
  segments[1].x1 = Eigen::Vector3d(0.5, 0.5, 0);
  segments[1].x2 = Eigen::Vector3d(0.5, 5.5, 0);
  uint32_t nb_keypoints = 10;
  double radii = 0.7;
  double scale = 0.03;

  return std::make_shared<Freeflyer3D>(
      "freeflyer_3", GetKeyPoints(nb_keypoints, segments),
      std::vector<double>(nb_keypoints, scale * radii));
}

//-----------------------------------------------------------------------------
// GetKeyPoints function implementation.
//-----------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------
// Freeflyer2D function implementation.
//-----------------------------------------------------------------------------

Freeflyer2D::~Freeflyer2D() {}
void Freeflyer2D::CreateTaskMaps(
    const std::vector<Eigen::VectorXd>& keypoints) {
  task_maps_.resize(keypoints.size());
  for (uint32_t i = 0; i < keypoints.size(); i++) {
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
    task_maps_[i] = std::make_shared<HomogeneousTransform3d>(keypoints[i]);
  }
}

}  // namespace bewego
