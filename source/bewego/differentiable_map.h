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
  DifferentiableMap() : debug_(false) {}

  virtual uint32_t output_dimension() const = 0;
  virtual uint32_t input_dimension() const = 0;

  /** Method called when call object */
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
    return Forward(x);
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

 protected:
  bool debug_;
};

/**
    f round g : f(g(q))

    This function should be called pullback if we approxiate
    higher order (i.e., hessians) derivaties by pullback, here it's
    computing the true 1st order derivative of the composition.
*/
class Compose : public DifferentiableMap {
 public:
  Compose(std::shared_ptr<const DifferentiableMap> f,
          std::shared_ptr<const DifferentiableMap> g) {
    // Make sure the composition makes sense
    assert(g->output_dimension() == f->input_dimension());
    f_ = f;
    g_ = g;
  }

  uint32_t output_dimension() const { return f_->output_dimension(); }
  uint32_t input_dimension() const { return g_->input_dimension(); }

  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const {
    return (*f_)((*g_)(q));
  }

  /**
      d/dq f(g(q)), applies chain rule.

            * J_f(g(q)) J_g

      If J is the jacobian of a function f(x), J_f = d/dx f(x)
        then the jacobian of the "pullback" of f defined on the
        range space of a map g, f(g(q)) is
                d/dq f(g(q)) = J_f(g(q)) J_g
        This method computes and
        returns this "pullback gradient" J_f (g(q)) J_g(q).
        WARNING: J_f is assumed to be a jacobian np.matrix object
    */
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const {
    return Evaluate(q).second;
  }

  /**
      d^2/dq^2 f(g(q)), applies chain rule.

            * J_g' H_f J_g + H_g J_f,

        https://en.wikipedia.org/wiki/Chain_rule (Higher derivatives)

        WARNING: If n > 1, where g : R^m -> R^n, we approximate the hessian
                 to the first term. This is equivalent to considering H_g = 0
                 It can be seen as operating a pullback of the curvature
                 tensor of f by g.

          https://en.wikipedia.org/wiki/Pullback_(differential_geometry)
    */
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& q) {
    assert(f_->output_dimension() == 1);
    auto x = (*g_)(q);
    auto J_g = g_->Jacobian(q);
    Eigen::MatrixXd H = J_g.transpose() * f_->Hessian(x) * J_g;
    if (g_->output_dimension() == 1) {
      H += f_->Jacobian(x) * Eigen::VectorXd::Ones(input_dimension()) *
           g_->Hessian(q);
    }
    return H;
  }

  // d/dq f(g(q)), applies chain rule.
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(
      const Eigen::VectorXd& q) const {
    auto g = g_->Evaluate(q);
    auto f_o_g = f_->Evaluate(g.first);
    return std::make_pair(f_o_g.first, f_o_g.second * g.second);
  }

 protected:
  std::shared_ptr<const DifferentiableMap> f_;
  std::shared_ptr<const DifferentiableMap> g_;
};

/**
 * Fill up the function_tests_ vector with test points
 * RunTests will check that the gradient and hessian are correctly
 * implemented against finte difference.
 */
class DifferentialMapTest : public testing::Test {
 public:
  DifferentialMapTest()
      : gradient_precision_(1e-6), hessian_precision_(1e-6), verbose_(false) {}

  /** \brief Returns true if implementation is the same as finite difference */
  void FiniteDifferenceTest(std::shared_ptr<const DifferentiableMap> phi,
                            const Eigen::VectorXd& x) const;

  /** \brief Run test procedure on all function tests in function_tests_. */
  void RunTests() const;

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

  double gradient_precision_;
  double hessian_precision_;
  bool verbose_;
};

using DifferentiableMapPtr = std::shared_ptr<const DifferentiableMap>;
using VectorOfMaps = std::vector<std::shared_ptr<const DifferentiableMap>>;

}  // namespace bewego