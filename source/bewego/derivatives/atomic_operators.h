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
  ZeroMap(uint32_t m, uint32_t n) : m_(m), n_(n) {}

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
  IdentityMap(uint32_t n) : dim_(n) {}

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

/** Simple map of the form: f(x) = ax + b */
class AffineMap : public DifferentiableMap {
 public:
  AffineMap(const Eigen::MatrixXd& a, const Eigen::VectorXd& b) : a_(a), b_(b) {
    assert(a_.rows() == b.size());
  }
  AffineMap(const Eigen::VectorXd& a, double b) {
    a_ = Eigen::MatrixXd(1, a.size());
    a_.row(0) = a;
    b_ = Eigen::VectorXd::Constant(1, b);
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
    return Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
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
  QuadricMap() {}
  QuadricMap(const Eigen::MatrixXd& a, const Eigen::VectorXd& b, double c)
      : a_(a), b_(b), c_(Eigen::VectorXd::Constant(1, c)) {
    assert(a_.rows() == a_.cols());
    assert(a_.rows() == b_.size());
    H_ = .5 * (a_ + a_.transpose());
  }

  QuadricMap(const Eigen::MatrixXd& a, const Eigen::VectorXd& b,
             const Eigen::VectorXd& c)
      : a_(a), b_(b), c_(c) {
    assert(c_.size() == 1);
    assert(a_.rows() == a_.cols());
    assert(a_.rows() == b_.size());
    H_ = .5 * (a_ + a_.transpose());
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

/** Simple squared norm: f(x)= 0.5 | x - x_0 | ^2 */
class SquaredNorm : public DifferentiableMap {
 public:
  SquaredNorm(uint32_t dim) : x0_(Eigen::VectorXd::Zero(dim)) {}
  SquaredNorm(const Eigen::VectorXd& x0) : x0_(x0) {}

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return x0_.size(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    Eigen::VectorXd delta_x = x - x0_;
    double d = 0.5 * delta_x.transpose() * delta_x;
    return Eigen::VectorXd::Constant(1, d);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    Eigen::MatrixXd J(1, input_dimension());
    J.row(0) = x - x0_;
    return J;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    assert(input_dimension() == x.size());
    return Eigen::MatrixXd::Identity(input_dimension(), input_dimension());
  }

 protected:
  Eigen::VectorXd x0_;
};

/** Test function that can be evaluated on a grid **/
class ExpTestFunction : public DifferentiableMap {
 public:
  ExpTestFunction() {}

  uint32_t output_dimension() const { return 1; }
  uint32_t input_dimension() const { return 2; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
    assert(q.size() == 2);
    Eigen::VectorXd v(1);
    v(0) = exp(-pow(2.0 * q[0], 2) - pow(0.5 * q[1], 2));
    return v;
  }
};

/**
    Takes only some outputs
    n is the input dimension, indices are the output
**/
class RangeSubspaceMap : public DifferentiableMap {
 public:
  RangeSubspaceMap(uint32_t n, const std::vector<uint32_t>& indices)
      : dim_(n), indices_(indices) {}

  uint32_t output_dimension() const { return indices_.size(); }
  uint32_t input_dimension() const { return dim_; }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    Eigen::VectorXd x_sub(indices_.size());
    for (int i = 0; i < x_sub.size(); i++) {
      x_sub[i] = x[indices_[i]];
    }
    return x_sub;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd I(Eigen::MatrixXd::Identity(dim_, dim_));
    Eigen::MatrixXd J(output_dimension(), input_dimension());
    for (int i = 0; i < J.rows(); i++) {
      J.row(i) = I.row(indices_[i]);
    }
    return J;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    return Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
  }

 protected:
  uint32_t dim_;
  std::vector<uint32_t> indices_;
};

class Scale : public DifferentiableMap {
 public:
  Scale(DifferentiableMapPtr f, double alpha) : f_(f), alpha_(alpha) {}

  uint32_t output_dimension() const { return f_->output_dimension(); }
  uint32_t input_dimension() const { return f_->input_dimension(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return alpha_ * f_->Forward(x);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return alpha_ * f_->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    return alpha_ * f_->Hessian(x);
  }

 protected:
  DifferentiableMapPtr f_;
  double alpha_;
};

class Offset : public DifferentiableMap {
 public:
  Offset(DifferentiableMapPtr f, const Eigen::VectorXd& offset)
      : f_(f), offset_(offset) {
    assert(offset_.size() == f_->output_dimension());
  }

  uint32_t output_dimension() const { return f_->output_dimension(); }
  uint32_t input_dimension() const { return f_->input_dimension(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    return f_->Forward(x) + offset_;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return f_->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    return f_->Hessian(x);
  }

 protected:
  DifferentiableMapPtr f_;
  Eigen::VectorXd offset_;
};

/**
 * \brief Represents the sum of a set of maps f_i.
 *
 *   y(x) = \sum_{i=1}^N f_i(x)
 */
class SumMap : public DifferentiableMap {
 public:
  SumMap() { maps_ = std::make_shared<VectorOfMaps>(); }
  SumMap(std::shared_ptr<const VectorOfMaps> maps) : maps_(maps) {
    assert(maps_->size() > 0);
    for (uint32_t i = 0; i < maps_->size(); i++) {
      assert(maps_->at(i)->input_dimension() == input_dimension());
      assert(maps_->at(i)->output_dimension() == output_dimension());
    }
  }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    Eigen::VectorXd y(Eigen::VectorXd::Zero(output_dimension()));
    for (uint32_t i = 0; i < maps_->size(); i++) {
      y += maps_->at(i)->Forward(x);
    }
    return y;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd J(
        Eigen::MatrixXd::Zero(output_dimension(), input_dimension()));
    for (uint32_t i = 0; i < maps_->size(); i++) {
      J += maps_->at(i)->Jacobian(x);
    }
    return J;
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    Eigen::MatrixXd H(
        Eigen::MatrixXd::Zero(input_dimension(), input_dimension()));
    for (uint32_t i = 0; i < maps_->size(); i++) {
      H += maps_->at(i)->Hessian(x);
    }
    return H;
  }

  const VectorOfMaps& terms() const { return (*maps_); }

  virtual uint32_t input_dimension() const {
    return maps_->back()->input_dimension();
  }
  virtual uint32_t output_dimension() const {
    return maps_->back()->output_dimension();
  }

 protected:
  std::shared_ptr<const VectorOfMaps> maps_;
};

/**
 * \brief Represents the sum of a set of maps f_i.
 *
 *   f(x) = g(x) h(x)
 */
class ProductMap : public DifferentiableMap {
 public:
  ProductMap(DifferentiableMapPtr f1, DifferentiableMapPtr f2)
      : g_(f1), h_(f2) {
    assert(f1->input_dimension() == f2->input_dimension());
    assert(f1->output_dimension() == 1);
    assert(f2->output_dimension() == 1);
  }

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    Eigen::VectorXd v1 = (*g_)(x);
    Eigen::VectorXd v2 = (*h_)(x);
    return v1 * v2;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    double v1 = (*g_)(x)[0];
    double v2 = (*h_)(x)[0];
    return v1 * h_->Jacobian(x) + v2 * g_->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    assert(x.size() == input_dimension());
    double v1 = (*g_)(x)[0];
    double v2 = (*h_)(x)[0];
    Eigen::MatrixXd J1 = g_->Jacobian(x);
    Eigen::MatrixXd J2 = h_->Jacobian(x);
    Eigen::MatrixXd H = v1 * h_->Hessian(x) + v2 * g_->Hessian(x);
    return H + J1.transpose() * J2 + J2.transpose() * J1;
  }

  virtual uint32_t input_dimension() const { return g_->input_dimension(); }
  virtual uint32_t output_dimension() const { return g_->output_dimension(); }

 protected:
  DifferentiableMapPtr g_;
  DifferentiableMapPtr h_;
};

// Represent a function as f(x) = argmin_i g_i(x).
// All functions g_i must be of the same input dimensionality,
// as specified during construction.
// WARNING: This operator may lead to discontunious derivatives
class Min : public DifferentiableMap {
 public:
  // All terms must be of dimension term_dimension.
  Min(uint32_t term_dimension) : term_dimension_(term_dimension) {}
  Min(const VectorOfMaps& v) { AddTerms(v); }
  virtual ~Min() {}

  void AddTerms(const VectorOfMaps& v) {
    assert(v.empty() != true);
    term_dimension_ = v.front()->input_dimension();
    for (auto& f : v) {
      assert(f->input_dimension() == term_dimension_);
      assert(f->output_dimension() == 1);
    }
    functions_ = v;
  }

  uint32_t GetMinFunctionId(const Eigen::VectorXd& x) const {
    double min = std::numeric_limits<double>::max();
    uint32_t min_id = 0;
    for (uint32_t i = 0; i < functions_.size(); i++) {
      double value = (*functions_[i])(x)[0];
      if (min > value) {
        min = value;
        min_id = i;
      }
    }
    return min_id;
  }

  // Evaluates f(x) = argmin_x (x).
  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    double min = std::numeric_limits<double>::max();
    for (auto& f : functions_) {
      double value = (*f)(x)[0];
      if (min > value) {
        min = value;
      }
    }
    return Eigen::VectorXd::Constant(1, min);
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return functions_[GetMinFunctionId(x)]->Jacobian(x);
  }

  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    return functions_[GetMinFunctionId(x)]->Hessian(x);
  }

  virtual uint32_t input_dimension() const { return term_dimension_; }
  virtual uint32_t output_dimension() const { return 1; }

  const VectorOfMaps& maps() const { return functions_; }

 protected:
  VectorOfMaps functions_;
  uint32_t term_dimension_;
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

}  // namespace bewego
