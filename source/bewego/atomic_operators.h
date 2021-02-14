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

#include <bewego/differentiable_map.h>

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
    Eigen::VectorXd v(1);
    v(0) = 0.5 * delta_x.transpose() * delta_x;
    return v;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    assert(input_dimension() == x.size());
    Eigen::MatrixXd J(output_dimension(), input_dimension());
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

/**
 * \brief Represents the sum of a set of maps f_i.
 *
 *   y(x) = \sum_{i=1}^N f_i(x)
 */
class SumMap : public DifferentiableMap {
 public:
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
    assert(f2->output_dimension() == 1);
    assert(f1->output_dimension() == 1);
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
    // TODO AND TEST.
    Eigen::MatrixXd H(1, 1);
    return H;
  }

  virtual uint32_t input_dimension() const { return g_->input_dimension(); }
  virtual uint32_t output_dimension() const { return g_->output_dimension(); }

 protected:
  DifferentiableMapPtr g_;
  DifferentiableMapPtr h_;
};

}  // namespace bewego
