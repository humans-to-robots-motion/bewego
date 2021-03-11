/**
 * Copyright (c) 2020, Jim Mainprice
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this
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
 *AND
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

#include <bewego/numerical_optimization/planar_motion_optimization.h>

// Eigen includes
#include <Eigen/Cholesky>
#include <Eigen/Core>

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace bewego::util;
using std::cout;
using std::endl;

namespace rieef {

PlanarOptimzer::PlanarOptimzer(uint32_t T, double dt,
                               const Rectangle& workspace_bounds)
    : MotionObjective(T, dt, 2),
      with_rotation_(false),
      with_attractor_constraint_(false) {
  cout << "Create planar optimizer with n : " << n_ << endl;
  assert(n_ == 2);
  assert(T > 2);
  assert(dt > 0);
}

void PlanarOptimzer::set_parameters_from_flags() {
  // verbose_ = FLAGS_verbose;
  // scalar_cspace_vel_ = FLAGS_cspace_vel_scalar;
  // scalar_cspace_acc_ = FLAGS_cspace_acc_scalar;
  // scalar_flow_ = FLAGS_scalar_flow;
  // scalar_geodesic_ = FLAGS_scalar_geodesic;
  // scalar_goal_constraint_ = FLAGS_scalar_goal_constraint;
  // scalar_obstacle_barrier_ = FLAGS_scalar_obstacle_barrier;
  // scalar_joint_limits_ = FLAGS_scalar_joint_limits;
  // scalar_surface_constraint_ = FLAGS_scalar_surface_constraint;
  // scalar_posture_ = FLAGS_scalar_posture;
  // visualize_inner_loop_ = FLAGS_visualize_inner_loop;
  // attractor_type_ = FLAGS_attractor_type;
}

PlanarOptimzer::FunctionNetwork PlanarOptimzer::GetEmptyFunctionNetwork()
    const {
  uint32_t input_dimension = CliquesFunctionNetwork::NetworkDim(T_, n_);
  return std::make_shared<CliquesFunctionNetwork>(input_dimension, T_);
}

std::vector<Bounds> PlanarOptimzer::GetJointLimits() const {
  assert(n_ == 2);
  std::vector<Bounds> limits(n_);
  auto extent = workspace_bounds_.extent();
  limits[0] = Bounds(extent.x_min(), extent.x_max());
  limits[1] = Bounds(extent.y_min(), extent.y_max());
  return limits;
}

std::vector<Bounds> PlanarOptimzer::GetDofBounds() const {
  // Joint limits on works for 2D freeflyer for now
  auto bounds = GetJointLimits();
  std::vector<Bounds> dof_bounds(bounds.size() * (T_ + 1));
  double joint_margin = .0;
  assert(bounds.size() == n_);
  assert(dof_bounds.size() == n_ * (T_ + 1));
  for (uint32_t t = 0; t <= T_; t++) {
    uint32_t d = 2;
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

void PlanarOptimzer::AddGoalConstraint(const Eigen::VectorXd& q_goal,
                                       double scalar) {
  auto goal_constraint_network = std::make_shared<>;
  auto terminal_potential =
      std::make_shared<Compose>(std::make_shared<SquaredNorm>(q_goal),
                                function_network_->CenterOfCliqueMap());
  function_network->RegisterFunctionForLastClique(
      std::make_shared<Scale>(terminal_potential, scalar));
}

std::shared_ptr<const ConstrainedOptimizer> PlanarOptimzer::SetupIpoptOptimizer(
    const Eigen::VectorXd& q_init) const {
  dstage("constructing optimizer");
  auto optimizer = std::make_shared<IpoptOptimizer>();
  if (FLAGS_ff_ipopt_with_bounds) {
    optimizer->set_bounds(GetDofBounds());
  }
  optimizer->set_verbose(verbose_);
  optimizer->set_option("print_level", verbose_ ? 4 : 0);
  optimizer->set_option("hessian_approximation",
                        FLAGS_ff_ipopt_hessian_approximation);
  // Parse all options from flags
  optimizer->SetFlagsOptions();

  visualizer_ = std::make_shared<FreeflyerOptimizationVisualizer>();
  stats_monitor_ = std::make_shared<rieef::StatsMonitor>();
  // Logging
  if (visualize_inner_loop_) {
    std::function<void(const Eigen::VectorXd&)> getter_function =
        std::bind(&FreeflyerOptimizationVisualizer::set_current_solution,
                  visualizer_.get(), std::placeholders::_1);
    optimizer->set_current_solution_accessor(getter_function);
    visualizer_->set_slow_down(FLAGS_visualize_slow_down);
    visualizer_->set_t_pause(FLAGS_visualize_t_pause);
    visualizer_->set_end_effector(end_effector_id_);
    visualizer_->InitializeFreeflyer("trajectory_array_3d", robot_->Clone(),
                                     q_init);
  }
  return optimizer;
}

// Returns the return value of the program.
std::shared_ptr<MarkovTrajectory> PlanarOptimzer::Optimize(
    const MarkovTrajectory& initial_traj, const Eigen::VectorXd& x_goal) const {
  std::shared_ptr<const ConstrainedOptimizer> optimizer;
  std::shared_ptr<const ConstrainedOptimizationProblem> nonlinear_problem;
  uint32_t T = initial_traj.T();
  uint32_t n = initial_traj.n();
  assert(n == n_);
  assert(T == T_);
  Eigen::VectorXd q_init = initial_traj.Configuration(0);
  dstage("Extracting constrained optimization problem");

  // setup networks
  auto objective_network = function_network_;

  auto f = TrajectoryOptimizationProblem::CreateDiffFunction(q_init,
                                                             objective_network);
  cout << "network input dim : " << f->input_dimension() << endl;

  // g_constraints.push_back(inequality_constraints_network);
  auto collision = GetKeyPointsSurfaceConstraints();
  for (auto& f : collision) {
    g_constraints.push_back(f);
  }
  if (attractor_constraint_) {
    auto network = GetEmptyFunctionNetwork();
    AddGoalConstraint(network, x_goal);
    h_constraints.push_back(network);
  }
  nonlinear_problem = std::make_shared<TrajectoryOptimizationProblem>(
      q_init, objective_network, g_constraints, h_constraints);
  optimizer = SetupIpoptOptimizer(q_init);

  // ---------------------------------------------------------------------------
  dstage("Optimizing trajectory");
  auto solution =
      optimizer->Optimize(*nonlinear_problem, initial_traj.ActiveSegment());
  auto optimal_trajectory =
      std::make_shared<Trajectory>(q_init, solution.x_opt());
  if (solution.warning_code() ==
      ConstrainedOptimizerSolution::DID_NOT_CONVERGE) {
    std::stringstream ss;
    printf("Did not converge! : %s", ss.str().c_str());
    // return std::shared_ptr<MarkovTrajectory>();
  } else {
    printf("Augmented lagrangian convered!");
  }
  if (verbose_ && visualize_inner_loop_) {
    sleep(1);
  }
  visualizer_->Stop();
  dstage("Optimization Done.");
  return optimal_trajectory;
}
}  // namespace rieef
