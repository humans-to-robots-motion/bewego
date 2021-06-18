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

#include <bewego/derivatives/combination_operators.h>
#include <bewego/workspace/collision_checking.h>

using namespace bewego;

// Creates a collision checker for a robot and a workspace
DifferentiableMapPtr ConstructCollisionChecker(
    const VectorOfCollisionPoints& collision_points,
    const VectorOfMaps& surface_functions, double margin) {
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
// SmoothCollisionConstraints function implementation.
//-----------------------------------------------------------------------------

SmoothCollisionConstraint::SmoothCollisionConstraint(
    const VectorOfMaps& surfaces, double gamma, double margin)
    : signed_distance_functions_(surfaces),
      margin_(margin),
      gamma_(gamma),
      f_(ConstructSmoothConstraint()) {
  type_ = "SmoothCollisionConstraint";
}

DifferentiableMapPtr SmoothCollisionConstraint::ConstructSmoothConstraint() {
  // Check input dimensions
  assert(signed_distance_functions_.size() > 0);
  // First iterate through all the surfaces in the environment
  // Add a constraint per keypoint on the freeflyer.
  // TODO have a different model for the robot (with capsules or ellipsoids)
  uint32_t n = signed_distance_functions_.size();
  if (margin_ != 0) {
    for (uint32_t i = 0; i < n; i++) {
      signed_distance_functions_[i] = (signed_distance_functions_[i] - margin_);
    }
  }
  auto smooth_min = std::make_shared<NegLogSumExp>(n, gamma_);
  auto stack = std::make_shared<CombinedOutputMap>(signed_distance_functions_);
  return ComposedWith(smooth_min, stack);
}

//-----------------------------------------------------------------------------
// SmoothCollisionConstraints function implementation.
//-----------------------------------------------------------------------------

SmoothCollisionPointsConstraint::SmoothCollisionPointsConstraint(
    const VectorOfCollisionPoints& collision_points,
    const VectorOfMaps& surfaces, double gamma)
    : collision_points_(collision_points),
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
  // Add a constraint per keypoint.
  // TODO have a different model for the robot (with capsules or ellipsoids)
  VectorOfMaps maps;
  for (auto& surface : surfaces_) {
    for (auto& sphere : collision_points_) {
      auto task_space =
          ComposedWith(surface - (sphere.radius + margin_), sphere.task_map);
      signed_distance_functions_.push_back(task_space);
      maps.push_back(task_space);
    }
  }
  if (!maps.empty()) {
    auto smooth_min = std::make_shared<NegLogSumExp>(maps.size(), gamma_);
    auto stack = std::make_shared<CombinedOutputMap>(maps);
    f_ = ComposedWith(smooth_min, stack);
  }
}

// Destructor.
SmoothCollisionPointsConstraint::~SmoothCollisionPointsConstraint() {}