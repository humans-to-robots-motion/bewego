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

#include <bewego/workspace.h>

#include <iostream>
using namespace bewego;
using std::cout;
using std::endl;

//------------------------------------------------------------------------------
// SphereDistance implementation.
//------------------------------------------------------------------------------

double SphereDistance::Evaluate(const Eigen::VectorXd& x) const {
  return (x - origin_).norm() - radius_;
}

double SphereDistance::Evaluate(const Eigen::VectorXd& x,
                                Eigen::VectorXd* g) const {
  Eigen::VectorXd x_d = x - origin_;
  double dist_origin = x_d.norm();
  *g = x_d / dist_origin;
  return dist_origin - radius_;
}

double SphereDistance::Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g,
                                Eigen::MatrixXd* H) const {
  // CHECK_NOTNULL(g);
  // CHECK_NOTNULL(H);
  // CHECK_EQ(x.rows(), input_dimension());
  uint32_t n = input_dimension();

  // Gradient.
  Eigen::VectorXd x_d = x - origin_;
  double dist_origin = x_d.norm();
  *g = x_d / dist_origin;

  // Hessian.
  /* cutoff curvature
   * d=0.1 -> (1/d)^3 = 1e3.
   * d=0.01 -> (1/d)^3 = 1e6
   */
  double d_inv = 1. / dist_origin;
  if (dist_ < std::numeric_limits<double>::max()) {
    d_inv = std::min(1. / dist_, d_inv);
  }
  auto I = Eigen::MatrixXd::Identity(n, n);
  *H = d_inv * I - std::pow(d_inv, 3) * x_d * x_d.transpose();

  // Value.
  return dist_origin - radius_;
}

//------------------------------------------------------------------------------
// RectangleDistance implementation.
//------------------------------------------------------------------------------

RectangleDistance::~RectangleDistance() {}

// this has to be in meters
// if we are inside the box we should return negative values
// if we are outside the box we should return postive values
double RectangleDistance::Evaluate(const Eigen::VectorXd& x) const {
  // CHECK_EQ(x.size(), input_dimension());
  // position in our coordinate system
  bool penetrates = true;
  double result_max_penetration = std::numeric_limits<double>::lowest();
  Eigen::VectorXd res = orientation_.transpose() * (x - center_);
  Eigen::VectorXd res_proj = res;
  for (uint32_t i = 0; i < res.size(); ++i) {
    double sign = res[i] > 0 ? 1. : -1.;
    double abs_res = sign * res[i];
    res_proj[i] = sign > 0 ? std::min(res[i], dimensions_[i])
                           : std::max(res[i], sign * dimensions_[i]);
    double dist = abs_res - dimensions_[i];
    if (penetrates && (dist < 0)) {
      if (dist > result_max_penetration) {
        result_max_penetration = dist;
      }
    } else {
      penetrates = false;
    }
  }
  // case outside of the rectangle
  if (!penetrates) {
    return (res - res_proj).norm();
  }
  return result_max_penetration;
}
double RectangleDistance::Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g,
                                   Eigen::MatrixXd* H) const {
  // CHECK_NOTNULL(g);
  // CHECK_NOTNULL(H);
  // CHECK_EQ(x.size(), input_dimension());
  // position in our coordinate system
  Eigen::VectorXd res = orientation_.transpose() * (x - center_);
  Eigen::VectorXd res_proj = res;
  Eigen::VectorXd sign(res.size());
  for (uint32_t i = 0; i < res.size(); i++) {
    sign[i] = res[i] > 0 ? 1. : -1.;
  }
  // this has to be in meters
  // if we are inside the box we should return negative values
  // if we are outside the box we should return postive values
  double result_max_penetration = std::numeric_limits<double>::lowest();
  uint32_t idx_max_penetration = 0;  // id of the face when closest to a face

  double result_min_distance = std::numeric_limits<double>::max();
  uint32_t idx_min_distance = 0;  // id of the face when closest to a face

  // find closest point on box
  bool penetrates = true;
  for (uint32_t i = 0; i < res.size(); ++i) {
    if (sign[i] > 0) {
      if (res[i] < dimensions_[i]) {
        res_proj[i] = res[i];
      } else {
        res_proj[i] = dimensions_[i];
        idx_min_distance = i;
      }
    } else {
      double abs_dim = sign[i] * dimensions_[i];
      if (res[i] > abs_dim) {
        res_proj[i] = res[i];
      } else {
        res_proj[i] = abs_dim;
        idx_min_distance = i;
      }
    }
    double dist = sign[i] * res[i] - dimensions_[i];
    if (penetrates && (dist < 0)) {
      if (dist > result_max_penetration) {
        result_max_penetration = dist;
        idx_max_penetration = i;
      }
    } else {
      penetrates = false;
    }
  }

  // case outside of the box
  if (!penetrates) {
    result_min_distance = (res - res_proj).norm();
    uint32_t nb_outerdimensions = dim_;
    uint32_t innerdimension = 0;
    for (uint32_t i = 0; i < res.size(); i++) {
      if (std::abs(res[i]) < dimensions_[i]) {
        nb_outerdimensions--;
        innerdimension = i;
      }
    }

    bool vertex = nb_outerdimensions == dim_;
    bool edge_3d = dim_ == 3 && nb_outerdimensions == 2;

    // case where the closest point is a vertex or edge in 3D case
    // see sphere distance.
    if (vertex || edge_3d) {
      Eigen::VectorXd x_d = x - (orientation_ * res_proj + center_);
      double d_inv = 1. / result_min_distance;
      auto I = Eigen::MatrixXd::Identity(input_dimension(), input_dimension());
      *g = d_inv * x_d;
      if (dist_cutoff_ < std::numeric_limits<double>::max()) {
        d_inv = std::min(1. / dist_cutoff_, d_inv);
      }
      *H = d_inv * I - std::pow(d_inv, 3) * x_d * x_d.transpose();

      // case where the closest point is on an edge
      // Simply kill the row of the hessian of the vertex case.
      // TODO only works with no orientation, probably should
      // remove the curvature in more clever way...
      if (edge_3d) {
        H->row(innerdimension) = Eigen::Vector3d::Zero();
      }
      return result_min_distance;
    }

    // case where the closest point is on the face of the box
    const double& s = sign[idx_min_distance];
    *g = s * orientation_ * axis_.col(idx_min_distance);
    *H = Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
    return result_min_distance;
  }

  // we have a penetration
  const double& s = sign[idx_max_penetration];
  *g = s * orientation_ * axis_.col(idx_max_penetration);
  *H = Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
  return result_max_penetration;
}

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
  return std::make_shared<RectangleDistance>(center_, dimensions_, orientation_,
                                             1e-2);
}
