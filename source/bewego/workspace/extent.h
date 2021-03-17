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
 *                                               Jim Mainprice Wed 10 Mar 2021
 */

#pragma once

namespace bewego {

// Struct to pass arguments around
// TODO change the API to use this
struct extent_t {
  extent_t() {}
  extent_t(const std::vector<double>& v) {
    assert(v.size() == 4 || v.size() == 6);
    for (uint32_t i = 0; i < (v.size() / 2); i++) {
      assert(v[2 * i + 1] > v[2 * i]);
    }
    extents_ = v;
  }
  extent_t(double x_min, double x_max, double y_min, double y_max) {
    extents_.resize(4);
    extents_[0] = x_min;
    extents_[1] = x_max;
    extents_[2] = y_min;
    extents_[3] = y_max;
  }
  extent_t(double x_min, double x_max, double y_min, double y_max, double z_min,
           double z_max) {
    extents_.resize(6);
    extents_[0] = x_min;
    extents_[1] = x_max;
    extents_[2] = y_min;
    extents_[3] = y_max;
    extents_[4] = z_min;
    extents_[5] = z_max;
  }
  extent_t(const extent_t& e) : extents_(e.extents_) {}

  double ExtendX() const { return x_max() - x_min(); }
  double ExtendY() const { return y_max() - y_min(); }
  double ExtendZ() const { return z_max() - z_min(); }

  Eigen::VectorXd Center() const {
    Eigen::VectorXd c(extents_.size() / 2);
    for (uint32_t i = 0; i < c.size(); i++) {
      c[i] = (extents_[2 * i + 1] + extents_[2 * i]) * .5;
    }
    return c;
  }

  void Expand(double factor) {
    double dl_x = ExtendX() * factor;
    double dl_y = ExtendY() * factor;
    extents_[0] -= dl_x;
    extents_[1] += dl_x;
    extents_[2] -= dl_y;
    extents_[3] += dl_y;
  }
  double x_min() const { return extents_[0]; }
  double x_max() const { return extents_[1]; }
  double y_min() const { return extents_[2]; }
  double y_max() const { return extents_[3]; }
  double z_min() const { return extents_[4]; }
  double z_max() const { return extents_[5]; }
  std::vector<double> extents_;

  // Output x_min, x_max, y_min, y_max to CSV format
  friend std::ostream& operator<<(std::ostream& os, const extent_t& v);
};

}  // namespace bewego