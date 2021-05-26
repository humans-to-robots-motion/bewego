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

#include <Eigen/Core>

namespace bewego {
namespace util {

//! variable bounds
struct Bounds {
  Bounds(double lower = 0.0, double upper = 0.0)
      : lower_(lower), upper_(upper) {}
  double lower_;
  double upper_;
};

//!\brief Clips vector x to bounds v
inline void BoundClip(const std::vector<Bounds>& bounds, Eigen::VectorXd* x) {
  assert(x);
  assert(x->size() == bounds.size());
  for (uint32_t i = 0; i < bounds.size(); i++) {
    (*x)[i] = std::max((*x)[i], bounds[i].lower_);
    (*x)[i] = std::min((*x)[i], bounds[i].upper_);
  }
}

}  // namespace util
}  // namespace bewego