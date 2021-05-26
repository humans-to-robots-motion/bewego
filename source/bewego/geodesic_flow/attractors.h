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

#pragma once

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/combination_operators.h>

namespace bewego {

DifferentiableMapPtr SoftNormOffset(const Eigen::VectorXd& x_goal,
                                    double soft_norm_alpha);

/**
 * \brief Natural Attractor
 *
 *   f(x) = | phi(x) - phi(x_g) |_2
 *
 *   where phi is the workspace_geometry_map. This attractor
 *   does not work when usin with non-convex geometries.
 *   Can be either squared or regular norm depending on a flag
 *   for IPOPT this is works the same, it handles internally the
 *   non-smooth ness of the regular norm at 0.
 */
class NaturalAttractor : public DifferentiableMap {
 public:
  NaturalAttractor(DifferentiableMapPtr workspace_geometry_map,
                   const Eigen::VectorXd& x, bool squared = false);
  virtual ~NaturalAttractor();
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;
  virtual uint32_t input_dimension() const { return f_->input_dimension(); }
  virtual uint32_t output_dimension() const { return 1; }

 protected:
  DifferentiableMapPtr f_;
  Eigen::VectorXd x_goal_;
  bool squared_potential_;
  bool soft_norm_;
  double attractor_soft_norm_alpha_;
};

/**
 * \brief Smooth Attractor
 *
 *   f(x) = sigma(x) * f(x) + (1-sigma(x)) * |x - x_g|^2
 *
 *   where sigma is a sigmoid function parameterized
 *   by the signed distance to the goal. This means that
 *   away from the goal f(x) is active and close to the goal
 *   this attractor transforms to the squared norm.
 *
 *   Note: The transition behave is set by selecting the
 *   transition distance and the interval in which the transition happens.
 *
 *   Inside we use a SphereDistanceWithValue because the value
 *   of the attractors (sphere or distance) do not match, which
 *   can generate a kink in the attractor.
 *
 *   This results in interpolating on gradient and hessian !
 *   TODO: rewrite test with new smoothness...
 */
class SmoothAttractor : public SmoothTransition {
 public:
  SmoothAttractor(DifferentiableMapPtr distance, const Eigen::VectorXd& x_goal,
                  double d_transition, double d_interval);
  virtual ~SmoothAttractor();

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  DifferentiableMapPtr distance() const { return f1_; }
  DifferentiableMapPtr smooth_distance() const { return f2_; }

 protected:
  double min_norm_;
  DifferentiableMapPtr euclidean_distance_;
};

/**
 * \brief Smooth Attractor
 *
 *   f(x) = sigma(x) * f(x) + (1-sigma(x)) * f_NAT(x)
 *
 *  TODO see smooth SmoothAttractor
 *
 *  The idea of the Smooth Natural is to blend a natural attractor
 *  with gradient field computed using a diffusion process
 *  and a value function comming from a analytic attractor such that
 *  at the goal the gradient vanishes and the hessian is proper conditioned.
 */
class SmoothNatural : public SmoothTransition {
 public:
  SmoothNatural(DifferentiableMapPtr attractor,
                DifferentiableMapPtr workspace_geometry_map,
                const Eigen::VectorXd& x_goal, double d_transition,
                double d_interval);
  virtual ~SmoothNatural();

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  DifferentiableMapPtr attractor() const { return f1_; }
  DifferentiableMapPtr smooth_attractor() const { return f2_; }

 protected:
  double min_norm_;
  DifferentiableMapPtr euclidean_distance_;
};

/**
 * \brief GeodesicDistance
 *
 *  A wrapper class that defines the value and gradient
 *  using separate objects. The hessian is set to identity, or using
 *  Jacobian of the gradient map.
 */
class GeodesicDistance : public DifferentiableMap {
 public:
  GeodesicDistance(DifferentiableMapPtr value,
                   DifferentiableMapPtr neg_gradient);
  virtual ~GeodesicDistance();

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  uint32_t input_dimension() const { return value_->input_dimension(); }
  uint32_t output_dimension() const { return 1; }

  // TODO this is dirty just for ROMAN paper
  // fix this by not having the value mutable
  void set_value(DifferentiableMapPtr v) const { value_ = v; }

 protected:
  mutable DifferentiableMapPtr value_;
  DifferentiableMapPtr neg_gradient_;
  bool gradient_from_value_;
  bool hessian_from_gradient_;
};

}  // namespace bewego
