/**
 * Copyright (c) 2021
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
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/numerical_optimization/ipopt_optimizer.h>
#include <bewego/numerical_optimization/planar_motion_optimization.h>

#include <Eigen/Core>

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
      ipopt_with_bounds_(true),
      ipopt_hessian_approximation_("limited-memory"),
      visualize_inner_loop_(false),
      visualize_slow_down_(false),
      visualize_t_pause_(100000) {
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

void PlanarOptimizer::set_trajectory_publisher(bool with_slow_down,
                                               uint32_t t_pause) {
  visualize_inner_loop_ = true;
  visualize_slow_down_ = with_slow_down;
  visualize_t_pause_ = t_pause;
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
  // auto d_goal = std::make_shared<SquaredNorm>(q_goal);
  auto d_goal = std::make_shared<SoftNorm>(.05, q_goal);
  auto phi = ComposedWith(d_goal, network->CenterOfCliqueMap());

  // Scale and register to a new network
  network->RegisterFunctionForLastClique(scalar * phi);
  h_constraints_.push_back(network);
}

void PlanarOptimizer::AddWayPointConstraint(const Eigen::VectorXd& q_waypoint,
                                            uint32_t t, double scalar) {
  uint32_t dim = function_network_->input_dimension();
  auto network = std::make_shared<FunctionNetwork>(dim, n_);

  auto d_waypoint = std::make_shared<SoftNorm>(.05, q_waypoint);
  auto phi = ComposedWith(d_waypoint, network->LeftMostOfCliqueMap());

  // Scale and register to a new network
  network->RegisterFunctionForClique(t, scalar * phi);
  h_constraints_.push_back(network);
}

void PlanarOptimizer::AddInequalityConstraintToEachActiveClique(
    DifferentiableMapPtr phi, double scalar) {
  // Scale and register to a new network
  // Set up surface constraints for key points.
  uint32_t dim = function_network_->input_dimension();
  for (uint32_t t = 0; t < T_; t++) {
    auto network = std::make_shared<FunctionNetwork>(dim, n_);
    network->RegisterFunctionForClique(t, dt_ * scalar * phi);
    g_constraints_.push_back(network);
  }
}

void PlanarOptimizer::AddSmoothKeyPointsSurfaceConstraints(double margin,
                                                           double gamma,
                                                           double scalar) {
  if (workspace_objects_.empty()) {
    cerr << "WARNING: no obstacles are in the workspace" << endl;
    return;
  }
  assert(function_network_.get() != nullptr);
  assert(n_ == 2);

  // Create clique constraint function phi
  auto surfaces = workspace_->ExtractSurfaceFunctions();
  auto sdf =
      std::make_shared<SmoothCollisionConstraints>(surfaces, gamma, margin);
  auto phi = ComposedWith(sdf, function_network_->CenterOfCliqueMap());

  AddInequalityConstraintToEachActiveClique(phi, scalar);
  // auto phi = TrajectoryConstraintNetwork(T_, n_, sdf, gamma);
  // g_constraints_unstructured_.push_back(scalar * phi);
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
  AddInequalityConstraintToEachActiveClique(phi, scalar);
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

// -----------------------------------------------------------------------------
// Optimization function
// -----------------------------------------------------------------------------
OptimizeResult PlanarOptimizer::Optimize(
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
  result.x = solution.x();
  result.fun = Eigen::VectorXd::Constant(1, solution.objective_value());
  result.message =
      solution.warning_code() == ConstrainedSolution::DID_NOT_CONVERGE
          ? "not converged"
          : "converged";
  return result;
}
