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

#include <bewego/derivatives/differentiable_map.h>

#include <vector>

namespace bewego {

/** Simple zero map : f(x) = 0 **/
class ZeroMap : public DifferentiableMap {
 public:
  ZeroMap(uint32_t m, uint32_t n) : m_(m), n_(n) { type_ = "ZeroMap"; }

  uint32_t output_dimension() const { return m_; }
  uint32_t input_dimension() const { return n_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(x.size() == n_);
    return Eigen::VectorXd::Zero(m_);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(x.size() == n_);
    return Eigen::MatrixXd::Zero(m_, n_);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(m_ == 1);
    assert(n_ == x.size());
    return Eigen::MatrixXd::Zero(n_, n_);
  }

 protected:
  uint32_t m_;
  uint32_t n_;
};

/** Simple identity map : f(x) = x **/
class IdentityMap : public DifferentiableMap {
 public:
  IdentityMap(uint32_t n) : dim_(n) { type_ = "IdentityMap"; }

  uint32_t output_dimension() const { return dim_; }
  uint32_t input_dimension() const { return dim_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return x;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return Eigen::MatrixXd::Identity(dim_, dim_);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    assert(input_dimension() == x.size());
    return Eigen::MatrixXd::Zero(dim_, dim_);
  }

 protected:
  uint32_t dim_;
};

/** Simple identity map : f(x) = 1/2 x^2 **/
class SquareMap : public DifferentiableMap {
 public:
  SquareMap() { type_ = "SquareMap"; }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return 1; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return .5 * x * x;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return x;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return Eigen::MatrixXd::Ones(1, 1);
  }
};

/** Simple map of the form: f(x) = ax + b */
class AffineMap : public DifferentiableMap {
 public:
  AffineMap(const Eigen::MatrixXd& a, const Eigen::VectorXd& b) : a_(a), b_(b) {
    assert(a_.rows() == b.size());
    PreAllocate();
    H_.setZero();
    type_ = "AffineMap";
  }
  AffineMap(const Eigen::VectorXd& a, double b) {
    a_ = Eigen::MatrixXd(1, a.size());
    a_.row(0) = a;
    b_ = Eigen::VectorXd::Constant(1, b);
    PreAllocate();
    H_.setZero();
    type_ = "AffineMap";
  }

  uint32_t output_dimension() const { return b_.size(); }
  uint32_t input_dimension() const { return a_.cols(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return a_ * x + b_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return a_;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    assert(input_dimension() == x.size());
    return H_;
  }

  const Eigen::MatrixXd& a() const { return a_; }
  const Eigen::VectorXd& b() const { return b_; }

 protected:
  Eigen::MatrixXd a_;
  Eigen::VectorXd b_;
};

/** Here we implement a quadric funciton of the form:
        f(x) = 1/2 x^T A x + bx + c */
class QuadricMap : public DifferentiableMap {
 public:
  QuadricMap() { type_ = "QuadricMap"; }
  QuadricMap(const Eigen::MatrixXd& a, const Eigen::VectorXd& b, double c)
      : a_(a), b_(b), c_(Eigen::VectorXd::Constant(1, c)) {
    assert(a_.rows() == a_.cols());
    assert(a_.rows() == b_.size());
    H_ = .5 * (a_ + a_.transpose());
    type_ = "QuadricMap";
  }

  QuadricMap(const Eigen::MatrixXd& a, const Eigen::VectorXd& b,
             const Eigen::VectorXd& c)
      : a_(a), b_(b), c_(c) {
    assert(c_.size() == 1);
    assert(a_.rows() == a_.cols());
    assert(a_.rows() == b_.size());
    H_ = .5 * (a_ + a_.transpose());
    type_ = "QuadricMap";
  }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return b_.size(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return .5 * x.transpose() * a_ * x + b_.transpose() * x + c_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return (H_ * x + b_).transpose();
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return H_;
  }

  QuadricMap& operator+=(const QuadricMap& f_rhs) {
    assert(H_.rows() == f_rhs.H_.rows());
    assert(a_.rows() == f_rhs.a_.rows());
    assert(H_.cols() == f_rhs.H_.cols());
    assert(b_.rows() == f_rhs.b_.rows());
    H_ += f_rhs.H_;
    a_ += f_rhs.a_;
    b_ += f_rhs.b_;
    c_ += f_rhs.c_;
    return *this;
  }

  const Eigen::MatrixXd& a() const { return a_; }
  const Eigen::VectorXd& b() const { return b_; }
  double c() const { return c_[0]; }

 protected:
  void Initialize(const Eigen::MatrixXd& a, const Eigen::VectorXd& b,
                  double c) {
    assert(a_.rows() == a_.cols());
    assert(a_.rows() == b_.size());
    a_ = a;
    b_ = b;
    c_ = Eigen::VectorXd::Constant(1, c);
    H_ = .5 * (a_ + a_.transpose());
  }
  Eigen::MatrixXd a_;
  Eigen::VectorXd b_;
  Eigen::VectorXd c_;
  Eigen::MatrixXd H_;
};

/**
 *   Second-order Taylor approximation of a differentiable map.
 *
 *          f(x) \approx f(x0) + g'(x-x0) + 1/2 (x-x0)'H(x-x0),
 *
 *   with g and H are the gradient and Hessian of f, respectively.
 *   TODO: Test.
 */
class SecondOrderTaylorApproximation : public QuadricMap {
 public:
  SecondOrderTaylorApproximation(const DifferentiableMap& f,
                                 const Eigen::VectorXd& x0);

 private:
  Eigen::VectorXd x0_;
  Eigen::VectorXd g0_;
  double fx0_;
};

/** Simple squared norm: f(x)= 0.5 | x - x_0 | ^2 */
class SquaredNorm : public DifferentiableMap {
 public:
  SquaredNorm(uint32_t dim) : x0_(Eigen::VectorXd::Zero(dim)) {
    PreAllocate();
    H_.setIdentity();
    type_ = "SquaredNorm";
  }
  SquaredNorm(const Eigen::VectorXd& x0) : x0_(x0) {
    PreAllocate();
    H_.setIdentity();
    type_ = "SquaredNorm";
  }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return x0_.size(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    y_[0] = 0.5 * (x - x0_).squaredNorm();
    return y_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    return (x - x0_).transpose();
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    assert(input_dimension() == x.size());
    return H_;
  }

 protected:
  Eigen::VectorXd x0_;
};

/** Test function that can be evaluated on a grid **/
class ExpTestFunction : public DifferentiableMap {
 public:
  ExpTestFunction() { type_ = "ExpTestFunction"; }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return 2; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
    assert(q.size() == 2);
    Eigen::VectorXd v(1);
    v(0) = exp(-pow(2.0 * q[0], 2) - pow(0.5 * q[1], 2));
    return v;
  }
};

/** Takes only some outputs
     - n : input dimension
     - indices.size() : output dimension
**/
class RangeSubspaceMap : public DifferentiableMap {
 public:
  RangeSubspaceMap(uint32_t n, const std::vector<uint32_t>& indices)
      : dim_(n), indices_(indices) {
    PreAllocate();
    PrealocateJacobian();
    PrealocateHessian();
    type_ = "RangeSubspaceMap";
  }

  uint32_t output_dimension() const { return indices_.size(); }
  uint32_t input_dimension() const { return dim_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    for (int i = 0; i < y_.size(); i++) {
      y_[i] = x[indices_[i]];
    }
    return y_;
  }
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == dim_);
    return J_;
  }
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    return H_;
  }

 protected:
  void PrealocateJacobian() {
    Eigen::MatrixXd I(Eigen::MatrixXd::Identity(dim_, dim_));
    J_ = Eigen::MatrixXd(output_dimension(), input_dimension());
    for (int i = 0; i < J_.rows(); i++) {
      J_.row(i) = I.row(indices_[i]);
    }
  }

  void PrealocateHessian() {
    if (output_dimension() == 1) {
      H_ = Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
    }
  }

  uint32_t dim_;
  std::vector<uint32_t> indices_;
};  // namespace bewego

/**
 *   Logarithmic Barrier
 *
 * Details:
 *
 *    f(x) = -log(x)
 *
 * for numerical stability
        when x < margin f(x) = \infty
 */
class LogBarrier : public DifferentiableMap {
 public:
  LogBarrier(double margin = 0) : margin_(margin) { type_ = "LogBarrier"; }
  virtual ~LogBarrier() {}

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return 1; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x_vect) const;
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x_vect) const;
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x_vect) const;

 protected:
  double margin_;
};

/** A smooth version of the norm function.
 *
 * Details:
 *
 *   f(x; \alpha) = sqrt(x^2 + \alpha^2) - \alpha
 *
 *   since equality constraints are squared, using squared
 *   norms makes the optimization unstable. The regular norm
 *   is not smooth.
 *
 *   Introduced by Tassa et al 2012 (IROS)
 */
class SoftNorm : public DifferentiableMap {
 public:
  SoftNorm(double alpha, uint32_t n)
      : n_(n),
        alpha_(alpha),
        alpha_sq_(alpha * alpha),
        x0_(Eigen::VectorXd::Zero(n)) {
    type_ = "SoftNorm";
  }
  SoftNorm(double alpha, const Eigen::VectorXd& x0)
      : n_(x0.size()), alpha_(alpha), alpha_sq_(alpha * alpha), x0_(x0) {
    type_ = "SoftNorm";
  }

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return n_; }

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

 protected:
  uint32_t n_;
  double alpha_;
  double alpha_sq_;
  Eigen::VectorXd x0_;
};

/*! \brief Implements a softmax between functions g_i.
 *
 * Details:
 *
 *   f(x; a) = 1/a log(sum_i e^{a g_i(x)})
 *
 * where 'a' is a constant scaling factor. Negative values of 'a' turn the
 * softmax into a soft min.
 */
class LogSumExp : public DifferentiableMap {
 public:
  LogSumExp(uint32_t n, double alpha = 1.)
      : n_(n), alpha_(alpha), inv_alpha_(1. / alpha) {
    type_ = "LogSumExp";
  }
  virtual ~LogSumExp() {}

  uint32_t input_dimension() const override { return n_; }
  uint32_t output_dimension() const override { return 1; }

  // Evaluates f(x) = 1/a log(sum_i e^{a g_i(x)}).
  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const override;
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const override;
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const override;

 protected:
  uint32_t n_;
  double alpha_;
  double inv_alpha_;
};

/*! \brief Implements a soft*min* between two functions f and h. Equivalent
 * to using a negative alpha.
 *
 * Details:
 *
 *   f(x; a) = log(sum_i e^{-a g_i(x)})
 *
 * where 'a' is a constant scaling factor. Negative values of 'a' turn the
 * softmin into a soft max.
 */
class NegLogSumExp : public LogSumExp {
 public:
  NegLogSumExp(uint32_t n, double alpha = 1.) : LogSumExp(n, -alpha) {
    type_ = "NegLogSumExp";
  }
  virtual ~NegLogSumExp();
};

}  // namespace bewego
