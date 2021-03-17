/**
 * Copyright (c) 2020, Jim Mainprice
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <bewego/motion/trajectory.h>
#include <bewego/numerical_optimization/constrained_optimization_problem.h>
#include <bewego/numerical_optimization/optimizer.h>
#include <gtest/gtest.h>

namespace bewego {
namespace numerical_optimization {

/*!\brief This version of the constrained optimization problem implements
 * a generic version of the trajectory optimization problem
 */
class TrajectoryOptimizationProblem
    : public OptimizationProblemWithConstraints {
 public:
  using FunctionNetworkPtr = std::shared_ptr<const CliquesFunctionNetwork>;

  TrajectoryOptimizationProblem(
      const Eigen::VectorXd& q_init, FunctionNetworkPtr objective,
      const std::vector<FunctionNetworkPtr>& inequality_constraints,
      const std::vector<FunctionNetworkPtr>& equality_constraints);

  static std::shared_ptr<TrajectoryObjectiveFunction> CreateDiffFunction(
      const Eigen::VectorXd& q_init, std::shared_ptr<CliquesFunctionNetwork> f);

 protected:
  Eigen::VectorXd q_init_;
};

/*!\brief Create a very basic trajectory objective, which first and second
 * order are exactly known convienient for testing against finite
 * differences This objective is a Quadratically Constrained Quadratic
 * Program (QCQP)
 *
 *        x^* = argmin_x [ 1/2 x^T A x ]
 *                s.t. 1/2| q_N - q_g |^2  = 0
 *
 * where A = K^T TrajectoryObjectiveTestK, is the matrix
 * computing accelerations by finite
 * differences along the trajectory. The objective is convex, it has
 * gradient is Ax and
 * hessian is A, since A is symetric (otherwise H = 1/2 (A + A^T).
 * q_g is the goal configuration, (quadric) constraint,
 * whose gradient is (q_N - q_g) and hessian is I.
 */
class TrajectoryObjectiveTest {
 public:
  using FunctionNetworkPtr = std::shared_ptr<CliquesFunctionNetwork>;

  TrajectoryObjectiveTest(uint32_t n, double dt, uint32_t T,
                          double scalar_cspace_acc,
                          double scalar_goal_constraint,
                          const Eigen::VectorXd& q_init,
                          const Eigen::VectorXd& q_goal);

  std::shared_ptr<TrajectoryObjectiveFunction> ObjectiveDiffFunction() const;
  std::shared_ptr<TrajectoryObjectiveFunction> EqualityConstraintDiffFunction()
      const;

  FunctionNetworkPtr f() const { return f_; }
  FunctionNetworkPtr h() const { return h_; }

 protected:
  // Setup the testing problem
  void SetUp();

  // Add terms
  void AddAccelerationNormTerms();
  void AddGoalPotentialTerm();

  bool verbose_;

  uint32_t n_;  // config space dim
  double dt_;   // time discretization
  uint32_t T_;  // horizon

  double scalar_cspace_acc_;       // weight acceleration
  double scalar_goal_constraint_;  // weight target

  Eigen::VectorXd q_init_;  // init configuration
  Eigen::VectorXd q_goal_;  // goal configuration

  FunctionNetworkPtr f_;  // objective function
  FunctionNetworkPtr h_;  // equality constraint
};

/*!\brief The TrajectoryOptimizationTest class
 * sets up a testing problem to evaluate a generic
 * trajectory optimization problem, where
 *
 *    x^* = argmin_x f(x)
 *            s.t. g(x) <= 0
 *                 h(x)  = 0
 *
 * where f, g and h are defined as FunctionNetworkPtr
 * an example instanciation is
 *
 *    x^* = argmin_x [ 1/2 x^T A x ]
 *                s.t. 1/2| q_N - q_g |^2  = 0
 *
 * where the objective and constraints are convex quadratics
 * that form a QCQP. (see TrajectoryObjectiveTest).
 */
class TrajectoryOptimizationTest : public testing::Test {
 public:
  using FunctionNetworkPtr = std::shared_ptr<const CliquesFunctionNetwork>;

  TrajectoryOptimizationTest() {}
  virtual ~TrajectoryOptimizationTest();

  virtual void SetUp() {}
  virtual void ValidateSolution(const ConstrainedSolution& solution) const {}

  bool verbose() const { return verbose_; }
  void set_verbose(bool v) { verbose_ = v; }

 protected:
  void SetupOptimizationProblem();
  bool verbose_;
  Eigen::VectorXd q_init_;                                  // config
  FunctionNetworkPtr objective_;                            // f
  std::vector<FunctionNetworkPtr> inequality_constraints_;  // g
  std::vector<FunctionNetworkPtr> equality_constraints_;    // h
  std::shared_ptr<const TrajectoryOptimizationProblem> nonlinear_problem_;
  Trajectory initial_solution_;
};

/*!\brief The TrajectoryQCQPTest class
 * sets up a testing problem to evaluate a generic
 * trajectory optimization problem, where the objective and constraints
 * are convex quadratics that form a QCQP. (see TrajectoryObjectiveTest).
 */
class TrajectoryQCQPTest : public TrajectoryOptimizationTest {
 public:
  TrajectoryQCQPTest() {}
  virtual ~TrajectoryQCQPTest();

  // Setup the testing problem
  virtual void SetUp();

  // Setsup the optimization objective
  virtual void ConstructObjective();

  // Compares the optimial solution to a
  // linear interpolation between q_init and q_goal
  virtual void ValidateSolution(const ConstrainedSolution& solution,
                                double tol = 1e-3);

 protected:
  Eigen::VectorXd q_goal_;
  uint32_t T_;
};

bool TestMotionOptimization();

}  // namespace numerical_optimization
}  // namespace bewego
