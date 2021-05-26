/*
 * Copyright (c) 2020
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
 *                                               Jim Mainprice Wed 4 Feb 2020
 */

// Free Flyer Optimizer
#include <bewego/derivatives/combination_operators.h>
#include <bewego/motion/differentiable_kinematics.h>
#include <bewego/numerical_optimization/bewopt_planar.h>
#include <bewego/numerical_optimization/freeflyer_optimization.h>
#include <bewego/numerical_optimization/ipopt_optimizer.h>

// Exernal includes
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace bewego::util;
using std::cerr;
using std::cout;
using std::endl;

FreeflyerOptimzer::FreeflyerOptimzer(uint32_t n, uint32_t T, double dt,
                                     ExtentBox workspace_bounds,
                                     std::shared_ptr<Freeflyer> robot)
    : TrajectoryOptimizer(T, dt, n),
      workspace_dim_(robot->n()),
      workspace_bounds_(workspace_bounds),
      robot_(robot),
      end_effector_id_(0),
      attractor_type_("euclidean") {
  cout << "Create freeflyer optimizer with n : " << n << endl;
  assert(n_ > 1);
  assert(T_ > 2);

  // Initialize to NULL
  geodesic_flow_ = DifferentiableMapPtr();

  // Initialize smooth collision constraint
  auto collision_checker = std::make_shared<FreeFlyerCollisionConstraints>(
      robot, workspace_->ExtractSurfaceFunctions());
  smooth_collision_constraint_ = collision_checker->smooth_constraint();
}

std::vector<util::Bounds> FreeflyerOptimzer::DofsBounds() const {
  assert(workspace_bounds_.dim() == workspace_dim_);
  std::vector<util::Bounds> limits(n_);
  for (uint32_t i = 0; i < workspace_dim_; i++) {
    limits[i].lower_ = workspace_bounds_.extent(i, ExtentBox::Min);
    limits[i].upper_ = workspace_bounds_.extent(i, ExtentBox::Max);
  }
  return limits;
}

DifferentiableMapPtr FreeflyerOptimzer::GetDistanceActivation() const {
  // First interate through all the key points in the environment
  // Add a constraint per keypoint on the robot.
  // TODO have a different model for the robot (with capsules or ellipsoids)
  VectorOfMaps sdfs;
  for (auto& sdf : workspace_->ExtractSurfaceFunctions()) {
    auto r = robot_->keypoint_radius(0);
    sdfs.push_back(sdf - r);
  }
  auto stack = std::make_shared<CombinedOutputMap>(sdfs);
  auto smooth_min =
      std::make_shared<NegLogSumExp>(sdfs.size(), freeflyer_gamma_);
  return LogisticActivation(ComposedWith(smooth_min, stack), freeflyer_k_);
}

void FreeflyerOptimzer::AddGeodesicFlowTerm(FunctionNetwork network) const {
  // Add term to agree with flow
  if (scalar_flow_ <= 0.) return;
  assert(geodesic_flow_.get() != nullptr);

  cout << "workspace_dim_ : " << workspace_dim_ << endl;

  auto position = std::make_shared<Position>(workspace_dim_);
  auto velocity = std::make_shared<Velocity>(workspace_dim_);
  auto normalize = std::make_shared<NormalizeMap>(workspace_dim_);

  // Normalized the flow and velocity
  auto n_vel = CombinedWith(normalize, velocity);
  auto n_flow = CombinedWith(normalize, CombinedWith(geodesic_flow_, position));
  // auto activation = Pullback(position, GetDistanceActivation());
  // auto sdf = Pullback(position, ConstructSignedDistanceField(workspace_));

  // auto range = std::make_shared<Arccos>();
  // Arccos has infinite derivative at the borders
  // An affine fit of arcos (y = ax + b)
  // a_2 = [1, arcos(1) = 0]
  // a_1 = [-1, arcos(-1) = pi]
  // a = (a_22 - a_12) / (a_21 - a_11) = -pi/2
  // b = arcos(0) = pi/2

  auto range = std::make_shared<AffineMap>(-M_PI / 2, M_PI / 2);
  // auto range = std::make_shared<Arccos>();
  auto angle = ComposedWith(std::make_shared<DotProduct>(n_flow, n_vel), range);
  // auto sq_angle = Compose(std::make_shared<ScalarSquaredDifference>(0),
  // angle); angle = std::make_shared<DefaultFunction>(angle, sdf); angle =
  // std::make_shared<ProductOfTwoFactors>(
  //      angle,
  //      LogisticActivation(
  //          Pullback(std::make_shared<Velocity>(workspace_dim_),
  //                   std::make_shared<SquaredNorm>(workspace_dim_)),
  //          FLAGS_freeflyer_k));

  // auto position_map = std::make_shared<Translation>(workspace_dim_);
  // auto position_map = robot_->keypoint_map(end_effector_id_);

  cout << "T : " << network->T() << endl;
  for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
    // Calculate PosVel of FK(q)
    auto positions = std::make_shared<MultiEvalMap>(
        robot_->keypoint_map(i), network->nb_clique_elements());
    auto posvel = std::make_shared<FiniteDifferencesPosVel>(n_, dt_);
    auto f = ComposedWith(angle, ComposedWith(posvel, positions));
    for (uint32_t t = 5; t <= T_ - 2; ++t) {
      network->RegisterFunctionForClique(t, f * dt_ * scalar_flow_);
    }
  }
}

void FreeflyerOptimzer::AddGeodesicTerm(FunctionNetwork network) const {
  // Penalize all dimensions equally, because the scaling of the dimensions
  // should already be handled by the workspace geometry map, itself.
  if (scalar_geodesic_ <= 0.) return;
  auto phi = workspace_->WorkspaceGeometryMap();
  Eigen::VectorXd regularizers = Eigen::VectorXd::Ones(phi->output_dimension());

  for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
    for (uint32_t t = 0; t <= T_ - 2; ++t) {
      // Penalization of velocity in
      // workspace geometry map (phi):
      //
      //        1/2 | d phi(q)|^2 = 1/2 | J_phi dx |^2

      auto potential = ComposedWith(
          std::make_shared<SquaredNorm>(n_),
          ComposedWith(
              std::make_shared<RangeSubspaceMap>(2 * n_, range(n_, 2 * n_)),
              std::make_shared<PosVelDifferentiableMap>(phi)));

      // Calculate PosVel of FK(q)
      auto positions = std::make_shared<MultiEvalMap>(
          robot_->keypoint_map(i), network->nb_clique_elements());
      auto posvel = std::make_shared<FiniteDifferencesPosVel>(n_, dt_);
      auto f = ComposedWith(potential, ComposedWith(posvel, positions));

      network->RegisterFunctionForClique(t, f * dt_ * scalar_geodesic_);
    }
  }
}

/*
  void FreeflyerOptimzer::AddInternalAddKeyPointBarriers(
      FunctionNetwork network) const {
    if (scalar_obstacle_barrier_ <= 0.) return;
    auto obstacle_barrier_shell = std::make_shared<ScalableRobustLogBarrier>(
        10000000., scalar_obstacle_barrier_);
    for (auto& surface_function : workspace_->ExtractSurfaceFunctions()) {
      VectorOfMaps env_barrier;  // Combined barrier in configuration space.

      for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
        auto keypoint = robot_->keypoint_map(i);
        auto radius = robot_->keypoint_radius(i);

        // Create clique constraint function phi
        auto sdf = workspace_->SignedDistanceField() - margin;
        auto phi = ComposedWith(sdf, function_network_->CenterOfCliqueMap());
      }

      network->RegisterSubCliqueFunctionForAllCliques(ComposedWith(
          std::make_shared<SumMap>(env_barrier),
  network->CenterOfCliqueMap()));
    }
  }


FreeflyerOptimzer::FunctionNetwork FreeflyerOptimzer::GetEmptyFunctionNetwork()
    const {
  auto network = std::make_shared<FixedSizedFunctionNetwork>(n_, T_);
  network->RegisterSubCliqueFunctionForAllCliques(
      kinematic_function_transformer_->PullPosTerm(
          std::make_shared<LinearFunction>(n_)));  // Zero function
  return network;
}

  */

std::vector<FreeflyerOptimzer::FunctionNetwork>
FreeflyerOptimzer::GetKeyPointsSurfaceConstraints() const {
  std::vector<FunctionNetwork> constraints;
  if (scalar_surface_constraint_ <= 0.) return constraints;
  if (clique_collision_constraints_) {
    // Set up surface constraints for key points.
    for (uint32_t t = 0; t < T_; t++) {
      auto network = std::make_shared<FunctionNetwork>(dim, n_);
      auto center_clique = function_network_->CenterOfCliqueMap();
      auto phi = ComposedWith(smooth_sdf_[t], center_clique);
      network->RegisterFunctionForClique(t, dt_ * scalar * phi);
      g_constraints_.push_back(network);
    }
  } else {
    // Set up surface constraints for key points.
    for (uint32_t t = 1; t <= T_; ++t) {
      for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
        auto p = robot_->keypoint_map(i);
        auto r = robot_->keypoint_radius(i);
        assert(p->input_dimension() == n_);
        for (auto& surface_function : workspace_->ExtractSurfaceFunctions()) {
          auto sdf =
              Scale(ShiftZero(surface_function, r), scalar_surface_constraint_);
          auto network = GetEmptyFunctionNetwork();
          network->RegisterTimeCenteredSubCliqueFunction(
              t, kinematic_function_transformer_->PullPosTerm(p, sdf));
          constraints.push_back(network);
        }
      }
    }
  }
  return constraints;
}

void PlanarOptimizer::AddKeyPointsSurfaceConstraints(double margin,
                                                     double scalar) {
  if (workspace_objects_.empty()) {
    cerr << "WARNING: no obstacles are in the workspace" << endl;
    return;
  }
  assert(function_network_.get() != nullptr);
  assert(n_ == 2);

  for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
    auto p = robot_->keypoint_map(i);
    auto r = robot_->keypoint_radius(i);
    // Create clique constraint function phi
    auto sdf = workspace_->SignedDistanceField() - r - margin;
    auto constraint_function = ComposedWith(sdf, p);
    auto center_clique = function_network_->CenterOfCliqueMap();
    auto phi = ComposedWith(constraint_function, center_clique);
    AddInequalityConstraintToEachActiveClique(phi, scalar);
  }
}

void FreeflyerOptimzer::AddPosturalTerms(double scalar) const {
  if (scalar_posture_ == 0.) return;
  auto default_q_potential =
      std::make_shared<SquaredNorm>(q_default_) * dt_ * scalar_posture_;
  for (uint32_t t = 0; t < T_; t++) {
    function_network_->RegisterTimeCenteredPositionFunction(
        t, default_q_potential);
  }
}

void FreeflyerOptimzer::AddGoalConstraint(const Eigen::VectorXd& x_goal,
                                          double scalar) const {
  if (scalar_goal_constraint_ <= 0) return;
  assert(x_goal.size() == n_);
  assert(robot_->n() == workspace_dim_);

  DifferentiableMapPtr attractor;
  DifferentiableMapPtr kinematic_map;

  cout << "ATTRACTOR TYPE : " << attractor_type_ << endl;

  if (attractor_type_ == "euclidean") {
    // Goalset
    kinematic_map = robot_->keypoint_map(end_effector_id_);
    Eigen::VectorXd p_goal = x_goal.segment(0, workspace_dim_);
    //    attractor = std::make_shared<SquaredNorm>(p_goal);
    attractor = SoftNormOffset(p_goal, .05);
  } else {
    // Goalset
    kinematic_map = robot_->keypoint_map(end_effector_id_);
    Eigen::VectorXd p_goal = x_goal.segment(0, workspace_dim_);
    if (attractor_type_ == "natural") {
      attractor = std::make_shared<NaturalAttractor>(
          workspace_->WorkspaceGeometryMap(), p_goal, true);
      // Use the following for softnorm !
      // Pullback(std::make_shared<ShiftMap>(-phi_g),
      //    NDim(std::make_shared<SoftNorm>(.025), phi_g.size()));
    }
    if (attractor_type_ == "geodesic") {
      double d_t = FLAGS_attractor_transition;  // where the transition happens
      double d_i = FLAGS_attractor_interval;    // how fast is the transition
      if (FLAGS_attractor_value_geodesic) {
        attractor = geodesic_distance_;
        if (FLAGS_attractor_make_smooth) {
          attractor = std::make_shared<SmoothAttractor>(
              ComposedWith(std::make_shared<ScalarSquaredDifference>(0),
                           geodesic_distance_),
              p_goal, d_t, d_i);
        }
      } else {
        // MAN IF THAT WORKS !!!!
        std::dynamic_pointer_cast<const GeodesicDistance>(geodesic_distance_)
            ->set_value(std::make_shared<NaturalAttractor>(
                workspace_->WorkspaceGeometryMap(), p_goal, true));
        if (FLAGS_attractor_make_smooth) {
          attractor = std::make_shared<SmoothNatural>(
              geodesic_distance_, workspace_->WorkspaceGeometryMap(), p_goal,
              d_t, d_i);
        }
      }
      // Square the distance
      // attractor =
      //   Compose(std::make_shared<ScalarSquaredDifference>(0), attractor);
    }
    if (attractor_type_ == "wasserstein") {
      // TODO
      // Square the distance
      attractor =
          ComposedWith(std::make_shared<ScalarSquaredDifference>(0), attractor);
    }
  }

  // Register
  attractor = attractor * scalar;

  // TODO remove.
  //  Eigen::VectorXd p_goal = x_goal.segment(0, workspace_dim_);
  //  attractor = std::make_shared<SquaredNorm>(p_goal);
  //  attractor =
  //      Pullback(std::make_shared<Translation>(workspace_dim_), attractor);

  if (kinematic_map) {
    attractor = Pullback(kinematic_map, attractor);
  }
  network->RegisterTimeCenteredPositionFunction(T_, attractor);
}

/**
std::shared_ptr<FixedSizedFunctionNetwork> FreeflyerOptimzer::ObjectiveNetwork(
    const Eigen::VectorXd& x_goal) const {
  auto objective_terms = std::make_shared<FixedSizedFunctionNetwork>(n_, T_);
  if (!FLAGS_attractor_constraint) {
    AddGoalConstraint(objective_terms, x_goal);
  }
  AddDerivativeNormTerms(objective_terms);
  AddGeodesicTerm(objective_terms);
  AddGeodesicFlowTerm(objective_terms);
  AddInternalAddKeyPointBarriers(objective_terms);
  AddPosturalTerms(objective_terms);
  return objective_terms;
}

std::shared_ptr<FixedSizedFunctionNetwork>
FreeflyerOptimzer::InequalityConstraints() const {
  auto constraints = std::make_shared<FixedSizedFunctionNetwork>(n_, T_);
  AddJointLimitConstraints(constraints);
  AddKeyPointsSurfaceConstraints(constraints);
  return constraints;
}

std::shared_ptr<FixedSizedFunctionNetwork>
FreeflyerOptimzer::EqualityConstraints(const Eigen::VectorXd& x_goal) const {
  // Add a potential at the goal for the final configuration on the path
  auto constraints = std::make_shared<FixedSizedFunctionNetwork>(n_, T_);
  return constraints;
}
*/

/**
std::shared_ptr<const lula::optimization::ConstrainedOptimizer>
FreeflyerOptimzer::SetupIpoptOptimizer(const Eigen::VectorXd& q_init) const {
  dstage("constructing optimizer");
  auto optimizer = std::make_shared<numerical_optimization::IpoptOptimizer>();
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
std::shared_ptr<MarkovTrajectory> FreeflyerOptimzer::Optimize(
    const MarkovTrajectory& initial_traj, const Eigen::VectorXd& x_goal) const {
  std::shared_ptr<const ConstrainedOptimizer> optimizer;
  std::shared_ptr<const ConstrainedOptimizationProblem> nonlinear_problem;
  uint32_t T = initial_traj.T();
  uint32_t n = initial_traj.n();
  CHECK_EQ(n, n_);
  CHECK_EQ(T, T_);
  Eigen::VectorXd q_init = initial_traj.Configuration(0);
  dstage("Extracting constrained optimization problem");

  // setup networks
  auto objective_network = ObjectiveNetwork(x_goal);
  auto inequality_constraints_network = InequalityConstraints();
  auto equality_constraints_network = EqualityConstraints(x_goal);

  auto f = CreateDiffFunction(q_init, objective_network);
  cout << "network input dim : " << f->input_dimension() << endl;

  using FunctionNetworkPtr =
      std::shared_ptr<const lula::motion::MarkovTrajectoryFunctionNetwork>;
  FunctionNetworkPtr f = objective_network;
  std::vector<FunctionNetworkPtr> g_constraints;
  std::vector<FunctionNetworkPtr> h_constraints;
  // g_constraints.push_back(inequality_constraints_network);
  auto collision = GetKeyPointsSurfaceConstraints();
  for (auto& f : collision) {
    g_constraints.push_back(f);
  }
  if (FLAGS_attractor_constraint) {
    auto network = GetEmptyFunctionNetwork();
    AddGoalConstraint(network, x_goal);
    h_constraints.push_back(network);
  }
  nonlinear_problem = std::make_shared<TrajectoryOptimizationProblem>(
      q_init, objective_network, g_constraints, h_constraints);
  optimizer = SetupIpoptOptimizer(q_init);

  dstage("Optimizing trajectory");
  lula::util::LabeledTimer timer;
  timer.Start("Time Optimization (ms)");
  auto solution =
      optimizer->Optimize(*nonlinear_problem, initial_traj.ActiveSegment());
  auto optimal_trajectory =
      std::make_shared<MarkovTrajectory>(q_init, solution.x_opt());
  if (solution.warning_code() ==
      ConstrainedOptimizer::Solution::DID_NOT_CONVERGE) {
    std::stringstream ss;
    ROS_WARN("Did not converge! : %s", ss.str().c_str());
    // return std::shared_ptr<MarkovTrajectory>();
  } else {
    ROS_INFO("Augmented lagrangian convered!");
  }
  if (verbose_ && visualize_inner_loop_) {
    sleep(1);
  }
  visualizer_->Stop();
  // stats_monitor_->Stop();
  timer.Stop();
  timer.PrintLabeledTimings();
  dstage("Optimization Done.");
  return optimal_trajectory;
}
*/
