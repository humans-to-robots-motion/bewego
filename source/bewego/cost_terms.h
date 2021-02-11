// Copyright (c) 2021, Universität Stuttgart.  All rights reserved.
// author: Jim Mainprice, mainprice@gmail.com

#pragma once

#include <bewego/atomic_operators.h>
#include <bewego/differentiable_map.h>
#include <bewego/util.h>

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
    a_ /= dt;
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
    return (*derivative_)(clique)*derivative_->a();
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
    derivative_ = std::make_shared<FiniteDifferencesVelocity>(dim, dt);
  }
};

}  // namespace bewego
