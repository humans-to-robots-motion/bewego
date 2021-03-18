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
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/util/range.h>
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

inline std::shared_ptr<SquaredNormVelocity> SquaredVelocityNorm(uint32_t n,
                                                                double dt) {
  return std::make_shared<SquaredNormVelocity>(n, dt);
}

inline std::shared_ptr<SquaredNormAcceleration> SquaredAccelerationNorm(
    uint32_t n, double dt) {
  return std::make_shared<SquaredNormAcceleration>(n, dt);
}

class ObstaclePotential : public DifferentiableMap {
 public:
  ObstaclePotential() {}
  ObstaclePotential(std::shared_ptr<const DifferentiableMap> sdf, double alpha,
                    double rho_scaling);
  virtual ~ObstaclePotential() {}

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  // Returns the dimension of the domain (input) space.
  virtual uint32_t input_dimension() const { return ambient_space_dim_; }
  virtual uint32_t output_dimension() const { return 1; }

  // Returns a pointer to the sdf
  std::shared_ptr<const DifferentiableMap> signed_distance_field() const {
    return signed_distance_field_;
  }

 private:
  uint32_t ambient_space_dim_;
  std::shared_ptr<const DifferentiableMap> signed_distance_field_;
  double alpha_;
  double rho_scaling_;
};

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

/**
 Collision constraint function that averages all collision points.
 Also provides an interface for dissociating each constraint
 The vector of collision points contains a pointer to each
 foward kinematics map (task map).
 to get the surfaces simply use workspace->ExtractSurfaceFunctions
 there is a surface per object in the workspace
 */
class SmoothCollisionConstraints : DifferentiableMap {
 public:
  SmoothCollisionConstraints(const VectorOfMaps& surfaces, double gamma);
  virtual ~SmoothCollisionConstraints();

  /**
   * @brief constraints
   * @return a vector of signed distance function defined over
   * configuration space. One for each collsion point
   */
  VectorOfMaps constraints() const { return signed_distance_functions_; }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return f_->input_dimension(); }

  /**
   * @brief smooth_constraint
   * @return a softmin function of the constraint
   */
  DifferentiableMapPtr smooth_constraint() const { return f_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return f_->Forward(x);
  }
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return f_->Jacobian(x);
  }
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    return f_->Hessian(x);
  }

 protected:
  double margin_;
  VectorOfMaps signed_distance_functions_;
  DifferentiableMapPtr f_;
  uint32_t n_;
  double gamma_;
};

}  // namespace bewego
