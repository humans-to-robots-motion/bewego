/*
 * Copyright (c) 2019
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
 *                                                              Wed 4 Feb 2019
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cassert>
#include <vector>

namespace bewego {

class DifferentiableMap {
 public:
  DifferentiableMap() : debug_(false), type_("Default"), is_atomic_(true) {}

  virtual uint32_t output_dimension() const = 0;
  virtual uint32_t input_dimension() const = 0;

  /** prealocates ouput structures on construction */
  void PreAllocate() {
    y_ = Eigen::VectorXd::Zero(output_dimension());
    g_ = Eigen::VectorXd::Zero(input_dimension());
    J_ = Eigen::MatrixXd::Zero(output_dimension(), input_dimension());
    H_ = Eigen::MatrixXd::Zero(input_dimension(), input_dimension());
  }

  /** Method called when call object */
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
    return Forward(x);
  }

  /** Should return an array or single value */
  virtual double ForwardFunc(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    return Forward(x)[0];
  }

  /** Should return an array or single value */
  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const = 0;

  /** Should return an array or single value
          n : input dimension
      Convienience function to get gradients
      in the same shape as the input vector
      for addition and substraction, of course gradients are
      only availables if the output dimension is one.
  */
  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    return Jacobian(x).row(0);
  }

  /** Should return a matrix or single value of
              m x n : ouput x input (dimensions)
          by default the method returns the finite difference jacobian.
          WARNING the object returned by this function is a numpy matrix.
  */
  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    return FiniteDifferenceJacobian(*this, x);
  }

  /** Should return the hessian matrix
              n x n : input x input (dimensions)
          by default the method returns the finite difference hessian
          that relies on the jacobian function.
          This method would be a third order tensor
          in the case of multiple output, we exclude this case for now.
          WARNING the object returned by this function is a numpy matrix.
          */
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    return FiniteDifferenceHessian(*this, x);
  }

  /** Evaluates the map and jacobian simultaneously. The default
          implementation simply calls both forward and Getjacobian()
          separately but overriding this method can make the evaluation
          more efficient
          */
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(
      const Eigen::VectorXd& x) const {
    return std::make_pair(Forward(x), Jacobian(x));
  }

  /** Evaluates the map and jacobian simultaneously. The default
          implementation simply calls both forward and Getjacobian()
          separately but overriding this method can make the evaluation
          more efficient
          */
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd> EvaluateAll(
      const Eigen::VectorXd& x) const {
    return std::make_tuple(Forward(x), Jacobian(x), Hessian(x));
  }

  /** Takes an object f that has a forward method returning
      a numpy array when querried.
      */
  static Eigen::MatrixXd FiniteDifferenceJacobian(const DifferentiableMap& f,
                                                  const Eigen::VectorXd& x);

  /** Takes an object f that has a forward method returning
      a numpy array when querried.
      */
  static Eigen::MatrixXd FiniteDifferenceHessian(const DifferentiableMap& f,
                                                 const Eigen::VectorXd& x);

  /** check against finite differences */
  bool CheckJacobian(double precision = 1e-12) const;

  /** check against finite differences */
  bool CheckHessian(double precision = 1e-12) const;

  /** Print check Jacobian and Hessian info */
  void set_debug(bool v = true) { debug_ = v; }

  /** Returns the type of the differentiable map */
  std::string type() const { return type_; }

  /** return true if it is an atomic an operator */
  bool is_atomic() const { return is_atomic_; }

  /** return true if it is the same operator */
  virtual bool Compare(const DifferentiableMap& other) const {
    return other.type_ == type_;
  }

 protected:
  bool debug_;
  mutable Eigen::VectorXd y_;
  mutable Eigen::VectorXd g_;
  mutable Eigen::MatrixXd J_;
  mutable Eigen::MatrixXd H_;
  std::string type_;
  bool is_atomic_;
};

/** Constructs a simple cache system that checks of the
    the function has been called with the same argument
    and returns the stored value when this is the case */
class CachedDifferentiableMap : public DifferentiableMap {
 public:
  CachedDifferentiableMap() { reset_cache(); }

  virtual uint32_t output_dimension() const = 0;
  virtual uint32_t input_dimension() const = 0;

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    if (y_is_cached_ && x == x_y_) return y_;
    x_y_ = x;
    y_ = Forward_(x);
    y_is_cached_ = true;
    return y_;
  }

  virtual Eigen::VectorXd Gradient(const Eigen::VectorXd& x) const {
    if (g_is_cached_ && x == x_g_) return g_;
    x_g_ = x;
    g_ = Gradient_(x);
    g_is_cached_ = true;
    return g_;
  }

  virtual Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    if (J_is_cached_ && x == x_J_) return J_;
    x_J_ = x;
    J_ = Jacobian_(x);
    J_is_cached_ = true;
    return J_;
  }

  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    if (H_is_cached_ && x == x_H_) return H_;
    x_H_ = x;
    H_ = Hessian_(x);
    H_is_cached_ = true;
    return H_;
  }

  virtual Eigen::VectorXd Forward_(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::MatrixXd Jacobian_(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::MatrixXd Hessian_(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::VectorXd Gradient_(const Eigen::VectorXd& x) const {
    assert(output_dimension() == 1);
    return Jacobian_(x).row(0);
  }

  void reset_cache() const {
    y_is_cached_ = false;
    g_is_cached_ = false;
    J_is_cached_ = false;
    H_is_cached_ = false;
  }

 protected:
  mutable Eigen::VectorXd x_y_;
  mutable Eigen::VectorXd x_g_;
  mutable Eigen::VectorXd x_J_;
  mutable Eigen::VectorXd x_H_;
  mutable bool y_is_cached_;
  mutable bool g_is_cached_;
  mutable bool J_is_cached_;
  mutable bool H_is_cached_;
};

using DifferentiableMapPtr = std::shared_ptr<const DifferentiableMap>;
using VectorOfMaps = std::vector<DifferentiableMapPtr>;

/** return true if it is the same operator */
inline bool operator==(DifferentiableMapPtr f, DifferentiableMapPtr g) {
  return f->Compare(*g);
}

/**
 * Fill up the function_tests_ vector with test points
 * RunTests will check that the gradient and hessian are correctly
 * implemented against finte difference.
 */
class DifferentialMapTest : public testing::Test {
 public:
  DifferentialMapTest(bool with_hesssian = true)
      : use_relative_eq_(false),
        gradient_precision_(1e-6),
        hessian_precision_(1e-6),
        verbose_(false) {}

  /* Returns true if implementation is the same as finite difference */
  void FiniteDifferenceTest(std::shared_ptr<const DifferentiableMap> phi,
                            const Eigen::VectorXd& x) const;

  virtual void SetUp() {}

  /* Run test procedure on all function tests in function_tests_. */
  void RunAllTests() const;

  /* Run test procedure on all function tests in function_tests_. */
  void AddRandomTests(std::shared_ptr<const DifferentiableMap> f, uint32_t n);

  bool verbose() const { return verbose_; }
  void set_verbose(bool v) { verbose_ = v; }
  void set_precisions(double gradient_precision, double hessian_precision) {
    gradient_precision_ = gradient_precision;
    hessian_precision_ = hessian_precision;
  }

 protected:
  std::vector<
      std::pair<std::shared_ptr<const DifferentiableMap>, Eigen::VectorXd>>
      function_tests_;

  bool use_relative_eq_;
  double gradient_precision_;
  double hessian_precision_;
  bool verbose_;
};

}  // namespace bewego