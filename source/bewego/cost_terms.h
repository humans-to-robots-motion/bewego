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

#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/util.h>
// #include <iostream>
// using std::cout;
// using std::endl;

namespace bewego {

/*!\brief Define velocities where clique = [ x_t ; x_{t+1} ] */
class FiniteDifferencesVelocity : public AffineMap {
 public:
  FiniteDifferencesVelocity(uint32_t dim, double dt)
      : AffineMap(Eigen::MatrixXd::Zero(dim, 2 * dim),
                  Eigen::VectorXd::Zero(dim)) {
    _InitializeMatrix(dim, dt);
  }

  /*!\brief Velocity = [ x_{t+1} - x_{t} ] / dt */
  void _InitializeMatrix(uint32_t dim, double dt) {
    auto identity = Eigen::MatrixXd::Identity(dim, dim);
    a_.block(0, 0, dim, dim) = -identity;
    a_.block(0, dim, dim, dim) = identity;
    a_ /= dt;
  }
};

/*!\brief Define accelerations where clique = [ x_{t-1} ; x_{t} ; x_{t+1} ] */
class FiniteDifferencesAcceleration : public AffineMap {
 public:
  FiniteDifferencesAcceleration(uint32_t dim, double dt)
      : AffineMap(Eigen::MatrixXd::Zero(dim, 3 * dim),
                  Eigen::VectorXd::Zero(dim)) {
    _InitializeMatrix(dim, dt);
  }

  /*!\brief Acceleration = [ x_{t+1} + x_{t-1} - 2 * x_{t} ] / dt^2 */
  void _InitializeMatrix(uint32_t dim, double dt) {
    auto identity = Eigen::MatrixXd::Identity(dim, dim);
    a_.block(0, 0, dim, dim) = identity;
    a_.block(0, dim, dim, dim) = -2 * identity;
    a_.block(0, 2 * dim, dim, dim) = identity;
    a_ /= (dt * dt);
  }
};

/*!\brief Define any norm of derivatives clique = [x_t ; x_{t+1} ; ... ] */
class SquaredNormDerivative : public DifferentiableMap {
 public:
  SquaredNormDerivative(uint32_t dim)
      : sq_norm_(std::make_shared<SquaredNorm>(Eigen::VectorXd::Zero(dim))) {}

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return derivative_->input_dimension(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& clique) const {
    return (*sq_norm_)((*derivative_)(clique));
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& clique) const {
    return (*derivative_)(clique).transpose() * derivative_->a();
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& clique) const {
    return derivative_->a().transpose() * derivative_->a();
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
  }
};

/** Defines SN of acceleration clique = [x_{t-1} ; x_{t} ; x_{t+1} ] */
class SquaredNormAcceleration : public SquaredNormDerivative {
 public:
  SquaredNormAcceleration(uint32_t dim, double dt)
      : SquaredNormDerivative(dim) {
    derivative_ = std::make_shared<FiniteDifferencesAcceleration>(dim, dt);
  }
};

}  // namespace bewego
