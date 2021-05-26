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
 *                                               Jim Mainprice Tue 10 Mar 2020
 */

#include <bewego/geodesic_flow/attractors.h>
#include <bewego/util/misc.h>
#include <bewego/workspace/workspace.h>

using namespace std;
using namespace bewego;
using namespace bewego::util;

// bool attractor_soft_norm, false
// double attractor_soft_norm_alpha, .05
// bool attractor_squared_potential, false
// bool gradient_from_value, false
// bool hessian_from_gradient, false

//-----------------------------------------------------------------------------
// SoftNorm function implementation.
//-----------------------------------------------------------------------------

DifferentiableMapPtr SoftNormOffset(const Eigen::VectorXd& x_goal,
                                    double soft_norm_alpha) {
  uint32_t n = x_goal.size();
  return ComposedWith(std::make_shared<SoftNorm>(soft_norm_alpha, n),
                      std::make_shared<IdentityMap>(n) - x_goal);
}

//-----------------------------------------------------------------------------
// NaturalAttractor function implementation.
//-----------------------------------------------------------------------------

NaturalAttractor::NaturalAttractor(DifferentiableMapPtr workspace_geometry_map,
                                   const Eigen::VectorXd& x, bool squared)
    : x_goal_(x) {
  soft_norm_ = false;
  squared_potential_ = squared;
  attractor_soft_norm_alpha_ = .05;
  if (soft_norm_) {
    f_ = ComposedWith(SoftNormOffset(workspace_geometry_map->Forward(x_goal_),
                                     attractor_soft_norm_alpha_),
                      workspace_geometry_map);
  } else if (squared_potential_) {
    f_ = ComposedWith(
        std::make_shared<SquaredNorm>(workspace_geometry_map->Forward(x_goal_)),
        workspace_geometry_map);
  } else {
    f_ = ComposedWith(std::make_shared<SphereDistance>(
                          workspace_geometry_map->Forward(x_goal_), 0),
                      workspace_geometry_map);
  }
}

NaturalAttractor::~NaturalAttractor() {}

Eigen::VectorXd NaturalAttractor::Forward(const Eigen::VectorXd& x) const {
  return f_->Forward(x);
}

Eigen::MatrixXd NaturalAttractor::Jacobian(const Eigen::VectorXd& x) const {
  return f_->Jacobian(x);
}

Eigen::MatrixXd NaturalAttractor::Hessian(const Eigen::VectorXd& x) const {
  return f_->Hessian(x);
}

//-----------------------------------------------------------------------------
// GeodesicFlowAttractor function implementation.
//-----------------------------------------------------------------------------

/*
 Evaluates the distance to an N-sphere
 WARNING: This function has been tested only for 2D and 3D cases.
 the hessian is still implemented with finite differences.

        f(x) = | x - x_o |^2

  This function uses a sphere distance for gradient and hessian
  with a different value for the distance.
*/
class SquaredNormWithValue : public SquaredNorm {
 public:
  // Constructor.
  SquaredNormWithValue(DifferentiableMapPtr f, double threshold,
                       const Eigen::VectorXd& origin, double radius,
                       double dist_cutoff = std::numeric_limits<double>::max())
      : SquaredNorm(origin), threshold_(threshold), f_(f) {}
  virtual ~SquaredNormWithValue() {}

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    y_ = SquaredNorm::Forward(x);
    return y_[0] < threshold_ ? y_ : f_->Forward(x);
  }
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    y_ = SquaredNorm::Forward(x);
    return y_[0] < threshold_ ? SquaredNorm::Jacobian(x) : f_->Jacobian(x);
  }
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    y_ = SquaredNorm::Forward(x);
    return y_[0] < threshold_ ? SquaredNorm::Hessian(x) : f_->Hessian(x);
  }

 protected:
  double threshold_;
  DifferentiableMapPtr f_;
};

// -----------------------------------------------------------------------------

SmoothAttractor::SmoothAttractor(DifferentiableMapPtr distance,
                                 const Eigen::VectorXd& x_goal,
                                 double d_transition, double d_interval)
    : SmoothTransition(
          distance,
          std::make_shared<SquaredNormWithValue>(
              distance, d_transition - d_interval / 2, x_goal, 0),
          std::make_shared<SphereDistance>(x_goal, d_transition, 1e-2),
          SmoothTransition::TemperatureParameter(d_interval)),
      euclidean_distance_(std::make_shared<SphereDistance>(x_goal, 0)) {}

SmoothAttractor::~SmoothAttractor() {}

Eigen::VectorXd SmoothAttractor::Forward(const Eigen::VectorXd& x) const {
  assert(n_ == x.size());
  return ActivationWeights::Forward(x);
}

Eigen::MatrixXd SmoothAttractor::Jacobian(const Eigen::VectorXd& x) const {
  // double v = ActivationWeights::Evaluate(x, g);
  //  if ((*euclidean_distance_)(x) > 1e-2) {
  //    double g_norm = (*g).norm();
  //    *g = *g / g_norm;
  //  }
  return ActivationWeights::Jacobian(x);
}

Eigen::MatrixXd SmoothAttractor::Hessian(const Eigen::VectorXd& x) const {
  // double v = ActivationWeights::Evaluate(x, g, H);
  //  if ((*euclidean_distance_)(x) > 1e-2) {
  // Normalize gradient
  //    double g_norm = (*g).norm();
  //    *g = *g / g_norm;

  // Pullback hessian through normalizer
  //    double dinv = 1. / g_norm;
  //    Eigen::MatrixXd M = (dinv * Eigen::VectorXd::Ones(n_)).asDiagonal();
  //    Eigen::MatrixXd H_norm = M - std::pow(dinv, 3) * x * x.transpose();
  //    *H = H_norm * (*H);
  //  }
  return ActivationWeights::Hessian(x);
}

//------------------------------------------------------------------------------
// SmoothNatural implementation.
//------------------------------------------------------------------------------

SmoothNatural::SmoothNatural(DifferentiableMapPtr attractor,
                             DifferentiableMapPtr workspace_geometry_map,
                             const Eigen::VectorXd& x_goal, double d_transition,
                             double d_interval)
    : SmoothTransition(
          attractor,
          std::make_shared<NaturalAttractor>(workspace_geometry_map, x_goal,
                                             true),
          std::make_shared<SphereDistance>(x_goal, d_transition, 1e-2),
          SmoothTransition::TemperatureParameter(d_interval)),
      euclidean_distance_(std::make_shared<SphereDistance>(x_goal, 0)) {
  min_norm_ = 5e-2;
}

SmoothNatural::~SmoothNatural() {}

Eigen::VectorXd SmoothNatural::Forward(const Eigen::VectorXd& x) const {
  assert(n_ == x.size());
  if ((*euclidean_distance_)(x)[0] < min_norm_) {
    return (*smooth_attractor())(x);
  }
  return ActivationWeights::Forward(x);
}

Eigen::MatrixXd SmoothNatural::Jacobian(const Eigen::VectorXd& x) const {
  if ((*euclidean_distance_)(x)[0] < min_norm_) {
    return smooth_attractor()->Jacobian(x);
  }
  return ActivationWeights::Jacobian(x);
}

Eigen::MatrixXd SmoothNatural::Hessian(const Eigen::VectorXd& x) const {
  if ((*euclidean_distance_)(x)[0] < min_norm_) {
    return smooth_attractor()->Hessian(x);
  }
  return ActivationWeights::Hessian(x);
}

//------------------------------------------------------------------------------
// GeodesicDistance implementation.
//------------------------------------------------------------------------------

GeodesicDistance::GeodesicDistance(DifferentiableMapPtr value,
                                   DifferentiableMapPtr neg_gradient)
    : value_(value),
      neg_gradient_(neg_gradient),
      gradient_from_value_(true),
      hessian_from_gradient_(false) {
  assert(value_->input_dimension() == neg_gradient_->input_dimension());
}

GeodesicDistance::~GeodesicDistance() {}

Eigen::VectorXd GeodesicDistance::Forward(const Eigen::VectorXd& x) const {
  assert(input_dimension() == x.size());
  return value_->Forward(x);
}

Eigen::MatrixXd GeodesicDistance::Jacobian(const Eigen::VectorXd& x) const {
  assert(input_dimension() == x.size());
  if (gradient_from_value_) {
    J_ = value_->Jacobian(x);
  } else {
    J_ = -1 * (*neg_gradient_)(x);
  }
  return J_;
}

Eigen::MatrixXd GeodesicDistance::Hessian(const Eigen::VectorXd& x) const {
  assert(input_dimension() == x.size());

  if (hessian_from_gradient_) {
    H_ = -1 * (*neg_gradient_).Jacobian(x);
  } else {
    H_ = Eigen::MatrixXd::Identity(input_dimension(), input_dimension());
  }
  return H_;
  ;
}
