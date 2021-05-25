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

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/derivatives/combination_operators.h>
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/util/range.h>
// #include <iostream>
// using std::cout;
// using std::endl;

namespace bewego {

/*! \brief Velocities
 *
 * Details:
 *
 *      f([x_t ; x_{t+1}]) = d/dt x_t
                           = ( x_{t+1} - x_{t} ) / dt
 */
class FiniteDifferencesVelocity : public LinearMap {
 public:
  FiniteDifferencesVelocity(uint32_t dim, double dt)
      : LinearMap(Eigen::MatrixXd::Zero(dim, 2 * dim)) {
    _InitializeMatrix(dim, dt);
    type_ = "FiniteDifferencesVelocity";
  }

  void _InitializeMatrix(uint32_t dim, double dt) {
    auto identity = Eigen::MatrixXd::Identity(dim, dim);
    // Ax = [-I , I][x_t ; x_{t+1}] / dt
    A_.block(0, 0, dim, dim) = -identity;
    A_.block(0, dim, dim, dim) = identity;
    A_ /= dt;
  }
};

/*! \brief Accelerations
 *
 * Details:
 *
 *      f([x_{t-1} ; x_t ; x_{t+1}]) = d^2/dt^2 x_t
 *                                   = ( x_{t+1} + x_{t-1} - 2 * x_{t} ) / dt^2
 */
class FiniteDifferencesAcceleration : public LinearMap {
 public:
  FiniteDifferencesAcceleration(uint32_t dim, double dt)
      : LinearMap(Eigen::MatrixXd::Zero(dim, 3 * dim)) {
    _InitializeMatrix(dim, dt);
    type_ = "FiniteDifferencesAcceleration";
  }

  void _InitializeMatrix(uint32_t dim, double dt) {
    auto identity = Eigen::MatrixXd::Identity(dim, dim);
    A_.block(0, 0, dim, dim) = identity;
    A_.block(0, dim, dim, dim) = -2 * identity;
    A_.block(0, 2 * dim, dim, dim) = identity;
    A_ /= (dt * dt);
  }
};

/*! \brief Position and Velocity
 *
 * Details:
 *
 *      f([x_t ; x_{t+1}]) = [x_t ; d/dt x_t]
 */
class FiniteDifferencesPosVel : public LinearMap {
 public:
  FiniteDifferencesPosVel(uint32_t dim, double dt)
      : LinearMap(Eigen::MatrixXd::Zero(2 * dim, 2 * dim)) {
    _InitializeMatrix(dim, dt);
    type_ = "FiniteDifferencesPosVel";
  }

  void _InitializeMatrix(uint32_t dim, double dt) {
    auto identity = Eigen::MatrixXd::Identity(dim, dim);

    // Ax =  [I,   0]
    //       [-I/dt , I/dt][x_t ; x_{t+1}]
    A_.block(0, 0, dim, dim) = identity;
    A_.block(dim, 0, dim, dim) = -identity / dt;
    A_.block(dim, dim, dim, dim) = identity / dt;
  }
};

/*!\brief Define sq norm of derivatives
 *
 * Details:
 *
 *   f(x_t) = 1/2 | d^(n)/dt x_t |^2
 *
 * clique = [x_t ; x_{t+1} ; ... ]
 */
class SquaredNormDerivative : public CombinationOperator {
 public:
  SquaredNormDerivative(uint32_t dim)
      : sq_norm_(std::make_shared<SquaredNorm>(Eigen::VectorXd::Zero(dim))) {
    type_ = "SquaredNormDerivative";
  }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return derivative_->input_dimension(); }

  virtual VectorOfMaps nested_operators() const {
    return VectorOfMaps({sq_norm_, derivative_});
  }
  virtual VectorOfMaps ouput_operators() const {
    return VectorOfMaps({sq_norm_});
  }
  virtual VectorOfMaps input_operators() const {
    return VectorOfMaps({derivative_});
  }

  Eigen::VectorXd Forward(const Eigen::VectorXd& clique) const {
    return (*sq_norm_)((*derivative_)(clique));
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& clique) const {
    return (*derivative_)(clique).transpose() * derivative_->A();
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& clique) const {
    return derivative_->A().transpose() * derivative_->A();
  }

 protected:
  DifferentiableMapPtr sq_norm_;
  std::shared_ptr<const AffineMap> derivative_;
};

/** Defines SN of velocities where clique = [x_t ; x_{t+1} ] */
class SquaredNormVelocity : public SquaredNormDerivative {
 public:
  SquaredNormVelocity(uint32_t dim, double dt) : SquaredNormDerivative(dim) {
    derivative_ = std::make_shared<FiniteDifferencesVelocity>(dim, dt);
    type_ = "SquaredNormVelocity";
  }
};

/** Defines SN of acceleration clique = [x_{t-1} ; x_{t} ; x_{t+1} ] */
class SquaredNormAcceleration : public SquaredNormDerivative {
 public:
  SquaredNormAcceleration(uint32_t dim, double dt)
      : SquaredNormDerivative(dim) {
    derivative_ = std::make_shared<FiniteDifferencesAcceleration>(dim, dt);
    type_ = "SquaredNormAcceleration";
  }
};

inline std::shared_ptr<SquaredNormVelocity> SquaredVelocityNorm(uint32_t n,
                                                                double dt) {
  return std::make_shared<SquaredNormVelocity>(n, dt);
}

inline std::shared_ptr<SquaredNormAcceleration> SquaredAccelerationNorm(
    uint32_t n, double dt) {
  return std::make_shared<SquaredNormAcceleration>(n, dt);
}

/** Barrier between values v_lower and v_upper */
class BoundBarrier : public DifferentiableMap {
 public:
  BoundBarrier(const Eigen::VectorXd& v_lower, const Eigen::VectorXd& v_upper)
      : v_lower_(v_lower), v_upper_(v_upper) {
    assert(v_lower.size() == v_upper.size());
    alpha_ = 1.;
    margin_ = 1e-10;

    // Warning: this does not work with the line search
    inf_ = std::numeric_limits<double>::max();
    type_ = "BoundBarrier";
  }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return v_lower_.size(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(x.size() == input_dimension());
    double value = 0.;
    for (uint32_t i = 0; i < x.size(); i++) {
      double x_i = x[i];
      double l_dist = x_i - v_lower_[i];
      double u_dist = v_upper_[i] - x_i;
      if (l_dist < margin_ || u_dist < margin_) {
        return Eigen::VectorXd::Constant(1, inf_);
      }

      // Log barrier f(x_i) = -log(d_u) + -log(d_l)
      value += -alpha_ * log(l_dist);
      value += -alpha_ * log(u_dist);
    }
    return Eigen::VectorXd::Constant(1, std::min(inf_, value));
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(x.size() == input_dimension());
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, input_dimension());
    for (uint32_t i = 0; i < x.size(); i++) {
      double x_i = x[i];
      double l_dist = x_i - v_lower_[i];
      double u_dist = v_upper_[i] - x_i;
      if (l_dist < margin_ || u_dist < margin_) {
        return Eigen::MatrixXd::Zero(1, input_dimension());
      }
      J(0, i) += -alpha_ / l_dist;
      J(0, i) += alpha_ / u_dist;
    }
    return J;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(x.size() == input_dimension());
    Eigen::MatrixXd H =
        Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
    for (uint32_t i = 0; i < x.size(); i++) {
      double x_i = x[i];
      double l_dist = x_i - v_lower_[i];
      double u_dist = v_upper_[i] - x_i;
      if (l_dist < margin_ || u_dist < margin_) {
        return Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
      }
      H(i, i) += alpha_ / (l_dist * l_dist);
      H(i, i) += alpha_ / (u_dist * u_dist);
    }
    return H;
  }

 protected:
  Eigen::VectorXd v_lower_;
  Eigen::VectorXd v_upper_;
  double margin_;
  double alpha_;
  double inf_;
};

}  // namespace bewego
