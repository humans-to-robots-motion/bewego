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

#include <bewego/numerical_optimization/ipopt_optimizer.h>
#include <bewego/numerical_optimization/planar_motion_optimization.h>

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace bewego::util;
using std::cerr;
using std::cout;
using std::endl;

// DEFINE_int32(ff_ipopt_maximum_iters, 200, "");
// DEFINE_string(ff_ipopt_hessian_approximation, "limited-memory", "");
// DEFINE_double(ff_ipopt_constr_viol_tol, 1e-7, "");
// DEFINE_double(ff_ipopt_acceptable_tol, 1e-6, "");
// DEFINE_double(ff_ipopt_tol, 1e-3, "");
// DEFINE_bool(ff_ipopt_with_bounds, true, "");

PlanarOptimizer::PlanarOptimizer(uint32_t T, double dt,
                                 const std::vector<double>& workspace_bounds)
    : MotionObjective(T, dt, 2),
      with_rotation_(false),
      with_attractor_constraint_(false),
      ipopt_with_bounds_(false),
      ipopt_hessian_approximation_("limited-memory") {
  cout << "Create planar optimizer with n : " << n_ << endl;
  assert(n_ == 2);
  assert(T > 2);
  assert(dt > 0);

  cout << "function_network_ : " << function_network_->input_dimension()
       << endl;

  verbose_ = true;

  // For now the workspace is axis-aligned
  assert(workspace_bounds.size() == 4);
  extent_t bounds(workspace_bounds);
  workspace_bounds_ = std::make_shared<Rectangle>(
      bounds.Center(), Eigen::Vector2d(bounds.ExtendX(), bounds.ExtendY()), 0);
}

std::vector<Bounds> PlanarOptimizer::DofBounds() const {
  assert(n_ == 2);
  std::vector<Bounds> limits(n_);
  auto extent = workspace_bounds_->extent();
  limits[0] = Bounds(extent.x_min(), extent.x_max());
  limits[1] = Bounds(extent.y_min(), extent.y_max());
  return limits;
}

std::vector<Bounds> PlanarOptimizer::TrajectoryDofBounds() const {
  // Joint limits with rotations are for freeflyers
  // the extension of the current optimizer should not be too hard
  // For now we use the case without rotations, where the configuration
  // dimension matches the workspace dimension
  assert(n_ == 2);
  assert(n_ == workspace_->dimension());
  auto bounds = DofBounds();
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

void PlanarOptimizer::AddGoalConstraint(const Eigen::VectorXd& q_goal,
                                        double scalar) {
  assert(function_network_.get() != nullptr);
  assert(n_ == 2);

  uint32_t dim = function_network_->input_dimension();
  auto network = std::make_shared<FunctionNetwork>(dim, n_);

  // Create clique constraint function phi
  auto d_goal = std::make_shared<SquaredNorm>(q_goal);
  auto phi = ComposedWith(d_goal, network->CenterOfCliqueMap());

  // Scale and register to a new network
  network->RegisterFunctionForLastClique(scalar * phi);
  h_constraints_.push_back(network);
}

void PlanarOptimizer::AddKeyPointsSurfaceConstraints(double margin,
                                                     double scalar) {
  if (workspace_objects_.empty()) {
    cerr << "WARNING: no obstacles are in the workspace" << endl;
    return;
  }
  assert(function_network_.get() != nullptr);
  assert(n_ == 2);

  // Create clique constraint function phi
  auto sdf = workspace_->SignedDistanceField() - margin;
  auto phi = ComposedWith(sdf, function_network_->CenterOfCliqueMap());

  // Scale and register to a new network
  uint32_t dim = function_network_->input_dimension();
  auto network = std::make_shared<FunctionNetwork>(dim, n_);
  network->RegisterFunctionForLastClique(scalar * phi);
  g_constraints_.push_back(network);
}

std::shared_ptr<const ConstrainedOptimizer>
PlanarOptimizer::SetupIpoptOptimizer(
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

  optimizer->set_option("constr_viol_tol", 1e-7);
  // optimizer->set_option("hessian_approximation",
  // ipopt_hessian_approximation_); Parse all options from flags
  // optimizer->set_options_map(ipopt_options);

  // Logging
  /* TODO
  visualizer_ = std::make_shared<FreeflyerOptimizationVisualizer>();
  stats_monitor_ = std::make_shared<rieef::StatsMonitor>();
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
  */
  return optimizer;
}

// -----------------------------------------------------------------------------
// Optimization function
// -----------------------------------------------------------------------------
Eigen::VectorXd PlanarOptimizer::Optimize(
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
  if(T != T_ || n != n_) {
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
  return solution.x();
}
