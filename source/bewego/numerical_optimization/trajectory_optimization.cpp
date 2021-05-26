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
#include <bewego/numerical_optimization/ipopt_optimizer.h>
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
  auto f =
      std::make_shared<TrajectoryObjectiveFunction>(q_init_, objective_network);
  hessian_sparcity_patern_ = f->HessianSparcityPatern();
  objective_function_ = f;

  // Convert inequality to dense
  inequality_constraints_.clear();
  for (auto& g : inequality_constraints_networks) {
    auto g_i = std::make_shared<TrajectoryObjectiveFunction>(q_init_, g);
    inequality_constraints_.push_back(g_i);
    g_gradient_sparcity_paterns_.push_back(g_i->gradient_sparcity_patern());
    assert(objective_function_->input_dimension() ==
           inequality_constraints_.back()->input_dimension());
  }

  // Convert equality to dense
  equality_constraints_.clear();
  for (auto& h : equality_constraints_networks) {
    auto h_i = std::make_shared<TrajectoryObjectiveFunction>(q_init_, h);
    equality_constraints_.push_back(h_i);
    h_gradient_sparcity_paterns_.push_back(h_i->gradient_sparcity_patern());
    assert(objective_function_->input_dimension() ==
           equality_constraints_.back()->input_dimension());
  }

  // Check that sizes don't not overflow
  n_g_ = size_t_to_uint(inequality_constraints_.size());
  n_h_ = size_t_to_uint(equality_constraints_.size());
}

//------------------------------------------------------------------------------
// TrajectoryOptimizer implementation.
//------------------------------------------------------------------------------

TrajectoryOptimizer::TrajectoryOptimizer(uint32_t T, double dt, uint32_t n)
    : MotionObjective(T, dt, n),
      with_rotation_(false),
      with_attractor_constraint_(false),
      ipopt_with_bounds_(true),
      ipopt_hessian_approximation_("limited-memory"),
      visualize_inner_loop_(false),
      visualize_slow_down_(false),
      visualize_t_pause_(100000) {}

void TrajectoryOptimizer::set_trajectory_publisher(bool with_slow_down,
                                                   uint32_t t_pause) {
  visualize_inner_loop_ = true;
  visualize_slow_down_ = with_slow_down;
  visualize_t_pause_ = t_pause;
}

std::vector<Bounds> TrajectoryOptimizer::TrajectoryDofBounds() const {
  // Joint limits with rotations are for freeflyers
  // the extension of the current optimizer should not be too hard
  // For now we use the case without rotations, where the configuration
  // dimension matches the workspace dimension
  assert(n_ == 2);
  assert(n_ == workspace_->dimension());
  auto bounds = DofsBounds();
  std::vector<Bounds> dof_bounds(bounds.size() * (T_ + 1));
  double joint_margin = .0;
  assert(bounds.size() == n_);
  assert(dof_bounds.size() == n_ * (T_ + 1));
  uint32_t d = workspace_->dimension();
  for (uint32_t t = 0; t <= T_; t++) {
    for (uint32_t i = 0; i < d; i++) {
      if (t > 0 && t < T_) {
        dof_bounds[t * n_ + i].upper_ = bounds[i].upper_ - joint_margin;
        dof_bounds[t * n_ + i].lower_ = bounds[i].lower_ + joint_margin;
      } else {
        dof_bounds[t * n_ + i].upper_ = std::numeric_limits<double>::max();
        dof_bounds[t * n_ + i].lower_ = std::numeric_limits<double>::lowest();
      }
    }
    if (with_rotation_) {
      uint32_t dof_rot = d == 2 ? 1 : 3;
      for (uint32_t i = 0; i < dof_rot; i++) {
        dof_bounds[t * n_ + d + i].upper_ = std::numeric_limits<double>::max();
        dof_bounds[t * n_ + d + i].lower_ =
            std::numeric_limits<double>::lowest();
      }
    }
  }
  return dof_bounds;
}

std::shared_ptr<const ConstrainedOptimizer>
TrajectoryOptimizer::SetupIpoptOptimizer(
    const Eigen::VectorXd& q_init,
    const std::map<std::string, double>& ipopt_options) const {
  auto optimizer = std::make_shared<IpoptOptimizer>();
  if (ipopt_with_bounds_) {
    optimizer->set_bounds(TrajectoryDofBounds());
  }
  optimizer->set_verbose(verbose_);
  optimizer->set_option("print_level", verbose_ ? 4 : 0);

  // optimizer->set_option("derivative_test", "first-order");
  // optimizer->set_option("derivative_test", "second-order");  // TODO remove
  // optimizer->set_option("derivative_test_tol", 1e-4);

  // optimizer->set_option("constr_viol_tol", 1e-7);
  // optimizer->set_option("hessian_approximation",
  // ipopt_hessian_approximation_); Parse all options from flags
  optimizer->set_options_map(ipopt_options);

  // Logging
  // stats_monitor_ = std::make_shared<StatsMonitor>();
  if (visualize_inner_loop_) {
    publisher_ = std::make_shared<TrajectoryPublisher>();
    if (visualize_slow_down_) {
      publisher_->set_slow_down(true);
      publisher_->set_t_pause(visualize_t_pause_);
    }
    std::function<void(const Eigen::VectorXd&)> getter_function =
        std::bind(&TrajectoryPublisher::set_current_solution, publisher_.get(),
                  std::placeholders::_1);
    optimizer->set_current_solution_accessor(getter_function);
    publisher_->Initialize("127.0.0.1", 5555, q_init);
  }
  return optimizer;
}

OptimizeResult TrajectoryOptimizer::Optimize(
    const Eigen::VectorXd& initial_traj_vect, const Eigen::VectorXd& x_goal,
    const std::map<std::string, double>& options) const {
  // 1) Get input data
  uint32_t dim = initial_traj_vect.size();
  Eigen::VectorXd q_init = initial_traj_vect.head(n_);
  Trajectory init_traj(q_init, initial_traj_vect.tail(dim - n_));
  uint32_t T = init_traj.T();
  uint32_t n = init_traj.n();
  assert(n == n_);
  assert(T == T_);
  if (T != T_ || n != n_) {
    // check consistency, TODO asserts are deactivated in pybind11, why?
    throw std::exception();
  }
  cout << "-- T : " << T << endl;
  cout << "-- verbose : " << verbose_ << endl;
  cout << "-- n_g : " << g_constraints_.size() << endl;
  cout << "-- n_h : " << h_constraints_.size() << endl;

  // 2) Create problem and optimizer
  auto nonlinear_problem = std::make_shared<TrajectoryOptimizationProblem>(
      q_init, function_network_, g_constraints_, h_constraints_);

  // 3) Optimize trajectory
  auto optimizer = SetupIpoptOptimizer(q_init, options);
  auto solution = optimizer->Run(*nonlinear_problem, init_traj.ActiveSegment());

  // 4) Return solution
  if (verbose_) {
    if (solution.warning_code() == ConstrainedSolution::DID_NOT_CONVERGE) {
      std::stringstream ss;
      printf("Did not converge! : %s", ss.str().c_str());
    } else {
      printf("Augmented lagrangian convered!");
    }
  }
  if (visualize_inner_loop_ && publisher_) {
    cout << "stop visulization..." << endl;
    publisher_->Stop();
  }
  OptimizeResult result;
  result.success = solution.success();
  result.x = solution.x();
  result.fun = Eigen::VectorXd::Constant(1, solution.objective_value());
  result.message =
      solution.warning_code() == ConstrainedSolution::DID_NOT_CONVERGE
          ? "not converged"
          : "converged";
  return result;
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
