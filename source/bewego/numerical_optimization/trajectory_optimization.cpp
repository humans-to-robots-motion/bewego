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

#include <bewego/motion/cost_terms.h>
#include <bewego/motion/trajectory.h>
#include <bewego/numerical_optimization/ipopt_problem.h>
#include <bewego/numerical_optimization/trajectory_optimization.h>
#include <bewego/util/misc.h>

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace bewego::util;
using std::cout;
using std::endl;

//------------------------------------------------------------------------------
// TrajectoryOptimizationProblem implementation.
//------------------------------------------------------------------------------

TrajectoryOptimizationProblem::TrajectoryOptimizationProblem(
    const Eigen::VectorXd& q_init, FunctionNetworkPtr objective_network,
    const std::vector<FunctionNetworkPtr>& inequality_constraints_networks,
    const std::vector<FunctionNetworkPtr>& equality_constraints_networks)
    : q_init_(q_init) {
  // Convert to objective functions
  objective_function_ =
      std::make_shared<TrajectoryObjectiveFunction>(q_init_, objective_network);

  // Convert inequality to dense
  inequality_constraints_.clear();
  for (auto& g : inequality_constraints_networks) {
    inequality_constraints_.push_back(
        std::make_shared<TrajectoryObjectiveFunction>(q_init_, g));
    assert(objective_function_->input_dimension() ==
           inequality_constraints_.back()->input_dimension());
  }

  // Convert equality to dense
  equality_constraints_.clear();
  for (auto& h : equality_constraints_networks) {
    equality_constraints_.push_back(
        std::make_shared<TrajectoryObjectiveFunction>(q_init_, h));
    assert(objective_function_->input_dimension() ==
           equality_constraints_.back()->input_dimension());
  }

  // Check that sizes don't not overflow
  n_g_ = size_t_to_uint(inequality_constraints_.size());
  n_h_ = size_t_to_uint(equality_constraints_.size());
}

//------------------------------------------------------------------------------
// TrajectoryObjectiveTest implementation.
//------------------------------------------------------------------------------

TrajectoryObjectiveTest::TrajectoryObjectiveTest(uint32_t n, double dt,
                                                 uint32_t T,
                                                 double scalar_cspace_acc,
                                                 double scalar_goal_constraint,
                                                 const Eigen::VectorXd& q_init,
                                                 const Eigen::VectorXd& q_goal)
    : n_(n),
      dt_(dt),
      T_(T),
      scalar_cspace_acc_(scalar_cspace_acc),
      scalar_goal_constraint_(scalar_goal_constraint),
      q_init_(q_init),
      q_goal_(q_goal) {
  SetUp();
}

// Create a difffunction given a
// Markov trajectory function network.
std::shared_ptr<TrajectoryObjectiveFunction>
TrajectoryOptimizationProblem::CreateDiffFunction(
    const Eigen::VectorXd& q_init, std::shared_ptr<CliquesFunctionNetwork> f) {
  return std::make_shared<TrajectoryObjectiveFunction>(q_init, f);
}

std::shared_ptr<TrajectoryObjectiveFunction>
TrajectoryObjectiveTest::ObjectiveDiffFunction() const {
  return TrajectoryOptimizationProblem::CreateDiffFunction(q_init_, f_);
}

std::shared_ptr<TrajectoryObjectiveFunction>
TrajectoryObjectiveTest::EqualityConstraintDiffFunction() const {
  return TrajectoryOptimizationProblem::CreateDiffFunction(q_init_, h_);
}

void TrajectoryObjectiveTest::SetUp() {
  // Initialize the kinematic transformer.
  f_ = std::make_shared<CliquesFunctionNetwork>((T_ + 2) * n_, n_);
  h_ = std::make_shared<CliquesFunctionNetwork>((T_ + 2) * n_, n_);
  AddAccelerationNormTerms();
  AddGoalPotentialTerm();
}

void TrajectoryObjectiveTest::AddAccelerationNormTerms() {
  if (scalar_cspace_acc_ <= 0) return;
  // Add acceleration potential to all cliques.
  auto derivative = std::make_shared<SquaredNormAcceleration>(n_, dt_);
  f_->RegisterFunctionForAllCliques(
      std::make_shared<Scale>(derivative, dt_ * scalar_cspace_acc_));
}

void TrajectoryObjectiveTest::AddGoalPotentialTerm() {
  if (scalar_goal_constraint_ <= 0) return;
  // Add goal potential to last clique.
  auto terminal_potential = std::make_shared<Compose>(
      std::make_shared<SquaredNorm>(q_goal_), h_->CenterOfCliqueMap());
  h_->RegisterFunctionForLastClique(
      std::make_shared<Scale>(terminal_potential, scalar_goal_constraint_));
}

//------------------------------------------------------------------------------
// TrajectoryOptimizationTest implementation.
//------------------------------------------------------------------------------

TrajectoryOptimizationTest::~TrajectoryOptimizationTest() {}

void TrajectoryOptimizationTest::SetupOptimizationProblem() {
  nonlinear_problem_ = std::make_shared<TrajectoryOptimizationProblem>(
      q_init_, objective_, inequality_constraints_, equality_constraints_);
}

//------------------------------------------------------------------------------
// TrajectoryQCQPTest implementation.
//------------------------------------------------------------------------------

TrajectoryQCQPTest::~TrajectoryQCQPTest() {}

void TrajectoryQCQPTest::SetUp() { ConstructObjective(); }
void TrajectoryQCQPTest::ConstructObjective() {
  // Simple trajectory optimization problem defined
  q_init_ = Eigen::Vector2d(0, 0);
  q_goal_ = Eigen::Vector2d(1, 1);
  T_ = 30;
  double dt = 0.01;
  uint32_t n = util::size_t_to_uint(q_init_.size());
  double scalar_acc = 1e-5;
  double scalar_goal = 1;
  auto problem = std::make_shared<TrajectoryObjectiveTest>(
      n, dt, T_, scalar_acc, scalar_goal, q_init_, q_goal_);
  inequality_constraints_.clear();
  equality_constraints_.clear();
  equality_constraints_.push_back(problem->h());
  objective_ = problem->f();

  // Setup the QCQP
  SetupOptimizationProblem();

  // Zero motion trajectory
  initial_solution_ = Trajectory(n, T_);
  for (uint32_t t = 0; t <= T_ + 1; ++t) {
    initial_solution_.Configuration(t) = q_init_;
  }
  cout << "Done configuring TrajectoryQCQPTest" << endl;
}

void TrajectoryQCQPTest::ValidateSolution(const ConstrainedSolution& solution,
                                          double tol) {
  Trajectory optimal_trajectory(q_init_, solution.x());
  Trajectory interpolated_trajectory =
      GetLinearInterpolation(q_init_, q_goal_, T_);
  ASSERT_EQ(T_, optimal_trajectory.T());
  for (uint32_t t = 0; t <= optimal_trajectory.T(); t++) {
    Eigen::VectorXd q_o = optimal_trajectory.Configuration(t);
    Eigen::VectorXd q_i = interpolated_trajectory.Configuration(t);
    ASSERT_EQ(q_o.size(), q_i.size());
    double max_diff = (q_o - q_i).cwiseAbs().maxCoeff();
    EXPECT_NEAR(max_diff, 0., tol);
    if (verbose_) {
      cout << "[" << t << "] q_o: " << q_o.transpose() << endl;
      cout << "[" << t << "] q_i: " << q_i.transpose() << endl;
      if (t == T_) {
        cout << "[" << t << "] q': " << q_o.transpose()
             << ", q_goal: " << q_goal_.transpose()
             << ", goal dist: " << (q_o - q_goal_).norm();
        cout << endl;
      }
    }
  }
}

bool bewego::numerical_optimization::TestMotionOptimization() {
  // Simple trajectory optimization problem defined
  Eigen::Vector2d q_init(0, 0);
  Eigen::Vector2d q_goal(1, 1);
  uint32_t n = util::size_t_to_uint(q_init.size());
  uint32_t T = 30;
  double dt = 0.01;
  double scalar_acc = 1e-5;
  double scalar_goal = 1;

  // Setup the QCQP
  auto problem_test = std::make_shared<TrajectoryObjectiveTest>(
      n, dt, T, scalar_acc, scalar_goal, q_init, q_goal);
  std::vector<std::shared_ptr<const CliquesFunctionNetwork>>
      inequality_constraints;
  std::vector<std::shared_ptr<const CliquesFunctionNetwork>>
      equality_constraints;
  equality_constraints.push_back(problem_test->h());
  auto objective = problem_test->f();
  auto nonlinear_problem = std::make_shared<TrajectoryOptimizationProblem>(
      q_init, objective, inequality_constraints, equality_constraints);

  // Zero motion trajectory
  auto initial_solution = InitializeZeroTrajectory(q_init, T);
  cout << "Done configuring Trajectory QCQP Test" << endl;

  bool verbose = false;
  for (uint32_t i = 0; i < 1; i++) {
    IpoptOptimizer constrained_optimizer;
    constrained_optimizer.set_verbose(verbose);
    // constrained_optimizer.set_option("derivative_test", "first-order");
    // constrained_optimizer.set_option("derivative_test", "second-order");
    // constrained_optimizer.set_option("derivative_test_tol", 1e-4);
    constrained_optimizer.set_option("constr_viol_tol", 1e-7);
    ConstrainedSolution solution = constrained_optimizer.Run(
        *nonlinear_problem, initial_solution->ActiveSegment());
    assert(solution.success());

    // Check that it is equal to linear interpolation
    Trajectory optimal_trajectory(q_init, solution.x());
    Trajectory interpolated_trajectory =
        GetLinearInterpolation(q_init, q_goal, T);
    assert(T_ == optimal_trajectory.T());
    for (uint32_t t = 0; t <= optimal_trajectory.T(); t++) {
      Eigen::VectorXd q_o = optimal_trajectory.Configuration(t);
      Eigen::VectorXd q_i = interpolated_trajectory.Configuration(t);
      assert(q_o.size() == q_i.size());
      double max_diff = (q_o - q_i).cwiseAbs().maxCoeff();
      assert(std::fabs(max_diff) < tol);
    }
  }
  return true;
}
