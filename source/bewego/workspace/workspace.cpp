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

#include <bewego/workspace/workspace.h>

#include <iostream>
using namespace bewego;
using std::cerr;
using std::cout;
using std::endl;

//------------------------------------------------------------------------------
// Circle implementation.
//------------------------------------------------------------------------------

Circle::~Circle() {}

DifferentiableMapPtr Circle::ConstraintFunction() const {
  return std::make_shared<SphereDistance>(center_, radius_);
}

//------------------------------------------------------------------------------
// Rectangle implementation.
//------------------------------------------------------------------------------

Rectangle::~Rectangle() {}

DifferentiableMapPtr Rectangle::ConstraintFunction() const {
  return std::make_shared<BoxDistance>(center_, dimensions_, orientation_,
                                       1e-2);
}

//-----------------------------------------------------------------------------
// SmoothCollisionConstraints function implementation.
//-----------------------------------------------------------------------------

SmoothCollisionConstraints::SmoothCollisionConstraints(
    const VectorOfMaps& surfaces, double gamma, double margin)
    : signed_distance_functions_(surfaces),
      margin_(margin),
      gamma_(gamma),
      f_(ConstructSmoothConstraint()) {
  type_ = "SmoothCollisionConstraints";
}

DifferentiableMapPtr SmoothCollisionConstraints::ConstructSmoothConstraint() {
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
