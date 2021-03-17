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
 *                                              Jim Mainprice We 18 Dec 2019
 */

#pragma once

#include <bewego/derivatives/atomic_operators.h>
#include <bewego/numerical_optimization/optimizer.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace bewego {
namespace numerical_optimization {

void ExpectNear(const Eigen::VectorXd& x, const Eigen::VectorXd& x_expected,
                double precision = 1e-6, bool verbose = false);

class QuadraticProgram : public OptimizationProblemWithConstraints {
 public:
  /*!\brief Construct a quadratic program specifying all elements.
   *
   * Form:
   *     min_x f(x)
   *     s.t. Ax <= a
   *          Bx  = b
   *
   * If there are no inequality or equality constraints, pass in empty
   * (uninitialized) parameters (A, a) or (B, b), respectively.
   */
  QuadraticProgram(const bewego::QuadricMap& f, const Eigen::MatrixXd& A,
                   const Eigen::VectorXd& a, const Eigen::MatrixXd& B,
                   const Eigen::VectorXd& b, bool use_selective_penalty = true)
      : f_(f),
        A_(A),
        a_(a),
        B_(B),
        b_(b),
        use_selective_penalty_(use_selective_penalty) {}

  double Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g_evaluations,
                  Eigen::VectorXd* h_evaluations) const override {
    if (num_inequality_constraints() > 0) {
      *g_evaluations = a_ - A_ * x;
    } else
      *g_evaluations = Eigen::VectorXd();

    if (num_equality_constraints() > 0) {
      *h_evaluations = b_ - B_ * x;
    } else
      *h_evaluations = Eigen::VectorXd();

    return f_.ForwardFunc(x);
  }

  uint32_t num_inequality_constraints() const override { return A_.rows(); }
  uint32_t num_equality_constraints() const override { return B_.rows(); }

 private:
  bewego::QuadricMap f_;

  // These objects implement the inequality constraints 'Ax \leq a' and the
  // equality constraints 'Bx = b'.
  Eigen::MatrixXd A_;
  Eigen::VectorXd a_;
  Eigen::MatrixXd B_;
  Eigen::VectorXd b_;

  // If true, use the selective penalty for inequality constraints.  Otherwise,
  // use the one-sided quadratic constraint.
  bool use_selective_penalty_;
};

/**
 * @brief The QuadricProgramOptimizationTest class
 * setsup a testing problem to evaluate a Quadric Program (QP) of the form
 *
 *    x^* = argmin_x [ x^t G x + c^t + d ]
 *            s.t. Ax <= a
 *                 Bx  = b
 */
class QuadricProgramOptimizationTest : public testing::Test {
 public:
  QuadricProgramOptimizationTest() {}

  // Setup the testing problem
  virtual void SetUp();

  // Use this solution validation for the linear one-sided quadratic inequality
  // penalties. This solution isn't actually as good as the one found by the
  // selective penalty term. (See below.)
  virtual void ValidateSolution(const ConstrainedSolution& solution) const;

  bool verbose() const { return verbose_; }
  void set_verbose(bool v) { verbose_ = v; }

 protected:
  bool verbose_;
  std::shared_ptr<const QuadraticProgram> qp_;
  std::shared_ptr<const QuadraticProgram> qp_sel_;
  std::shared_ptr<const QuadricMap> f_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd a_;
  Eigen::MatrixXd B_;
  Eigen::VectorXd b_;
  Eigen::VectorXd x0_;
};

/**
 * @brief The GenericQuadricProgramTest class
 * Takes the QP and puts in the generic form:
 *
 *    x^* = argmin_x [ x^t G x + c^t + d ]
 *            s.t. g(x) : -Ax + a > 0  # inequality cnstraints
 *                 h(x) : -Bx + b = 0  # equality constraints
 */
class GenericQuadricProgramTest : public QuadricProgramOptimizationTest {
 public:
  GenericQuadricProgramTest() : QuadricProgramOptimizationTest() {}

  // Setup the testing problem
  virtual void SetUp();

  // Use this solution validation for the linear one-sided quadratic inequality
  // penalties. This solution isn't actually as good as the one found by the
  // selective penalty term. (See below.)
  virtual void ValidateSolution(const ConstrainedSolution& solution) const;

 protected:
  std::shared_ptr<const OptimizationProblemWithConstraints> nonlinear_problem_;
  std::vector<DifferentiableMapPtr> g_constraints_;
  std::vector<DifferentiableMapPtr> h_constraints_;
  uint32_t n_g_;  // Number of inequalities
  uint32_t n_h_;  // Number of equalities
};

/**
 * @brief The QuadricalyConstrainedQuadricProgramOptimizationTest class
 * this testing problem evaluates a QCQP of the form:
 *
 *    x^* = argmin_x [ x^t G x + c^t + d ]
 *            s.t. x^t A x + a_0^t x + a_1 <= a
 *                 Bx  = b
 */
class GenericQuadricalyConstrainedQuadricProgramTest
    : public GenericQuadricProgramTest {
 public:
  GenericQuadricalyConstrainedQuadricProgramTest()
      : GenericQuadricProgramTest() {}

  // Setup the testing problem
  virtual void SetUp();

  // Use this solution validation for the linear one-sided quadratic inequality
  // penalties. This solution isn't actually as good as the one found by the
  // selective penalty term. (See below.)
  virtual void ValidateSolution(const ConstrainedSolution& solution) const;

 protected:
  std::shared_ptr<const QuadricMap> g_;
  Eigen::VectorXd a_;
};

}  // namespace numerical_optimization
}  // namespace bewego
