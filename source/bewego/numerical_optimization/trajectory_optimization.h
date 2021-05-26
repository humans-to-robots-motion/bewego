/**
 * Copyright (c) 2020, Jim Mainprice
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
 *                                                             Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com
#pragma once

#include <bewego/motion/objective.h>
#include <bewego/motion/publisher.h>
#include <bewego/motion/trajectory.h>
#include <bewego/numerical_optimization/constrained_optimization_problem.h>
#include <bewego/numerical_optimization/optimizer.h>
#include <bewego/util/misc.h>
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

/*!@brief This version of the constrained trajectory optimizer
 * optimizes a TrajectoryOptimizationProblem using IPOPT so far.
 * TODO test class...
 */
class TrajectoryOptimizer : public MotionObjective {
 public:
  TrajectoryOptimizer(uint32_t T,  // number of cliques
                      double dt,   // time between cliques
                      uint32_t n   // config space dim
  );

  /** @brief Optimize a given trajectory */
  OptimizeResult Optimize(
      const Eigen::VectorXd& initial_traj,          // entire trajectory
      const Eigen::VectorXd& x_goal,                // goal configuration
      const std::map<std::string, double>& options  // optimizer options
  ) const;

  /** @brief Adds trajectory publisher (t_pause in microseconds) */
  void set_trajectory_publisher(bool with_slow_down, uint32_t t_pause = 100000);

  /** @brief Adds goal constraint */
  void AddInequalityConstraintToEachActiveClique(DifferentiableMapPtr phi,
                                                 double scalar);

  /** @brief Adds goal constraint */
  void AddGoalConstraint(const Eigen::VectorXd& q_goal, double scalar);

  /** @brief Adds waypoint constraint */
  void AddWayPointConstraint(const Eigen::VectorXd& q_waypoint, uint32_t t,
                             double scalar);

 protected:
  std::shared_ptr<const ConstrainedOptimizer> SetupIpoptOptimizer(
      const Eigen::VectorXd& q_init,
      const std::map<std::string, double>& ipopt_options) const;

  virtual std::vector<util::Bounds> DofsBounds()
      const = 0;  // Dof bounds limits
  std::vector<util::Bounds> TrajectoryDofBounds()
      const;  // Dof bounds trajectory

  typedef CliquesFunctionNetwork FunctionNetwork;
  typedef std::shared_ptr<const FunctionNetwork> FunctionNetworkPtr;
  typedef std::shared_ptr<const DifferentiableMap> ElementaryFunction;

  // options
  bool with_rotation_;
  bool with_attractor_constraint_;
  bool ipopt_with_bounds_;
  std::string ipopt_hessian_approximation_;

  // constraints networks
  std::vector<DifferentiableMapPtr>
      g_constraints_unstructured_;                 // inequalities
  std::vector<FunctionNetworkPtr> g_constraints_;  // inequalities
  std::vector<FunctionNetworkPtr> h_constraints_;  // equalities

  // logging
  bool visualize_inner_loop_;
  bool visualize_slow_down_;
  uint32_t visualize_t_pause_;
  bool monitor_inner_statistics_;
  mutable std::shared_ptr<TrajectoryPublisher> publisher_;
  // visualizer_; mutable std::shared_ptr<util::StatsMonitor> stats_monitor_;
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
