// Copyright (c) 2019, Universität Stuttgart.  All rights reserved.
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

 protected:
  Eigen::MatrixXd a_;
  Eigen::VectorXd b_;
};

/** Simple squared norm: f(x)= 0.5 | x - x_0 | ^2 */
class SquaredNorm : public DifferentiableMap {
 public:
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
class RangeSubspace : public DifferentiableMap {
 public:
  RangeSubspace(uint32_t n, const std::vector<uint32_t>& indices)
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
class Sum : public DifferentiableMap {
 public:
  Sum(std::shared_ptr<const VectorOfMaps> maps) : maps_(maps) {
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

  DifferentiableMapPtr map(uint32_t i) const { return (*maps_)[i]; }
  uint32_t num_maps() const { return maps_->size(); }

  virtual uint32_t input_dimension() const {
    return maps_->back()->input_dimension();
  }
  virtual uint32_t output_dimension() const {
    return maps_->back()->output_dimension();
  }

 protected:
  std::shared_ptr<const VectorOfMaps> maps_;
};

}  // namespace bewego
