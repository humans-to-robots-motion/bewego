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
 *                                              Jim Mainprice Mon 14 June 2021
 */

// Robot Optimizer
#include <bewego/derivatives/combination_operators.h>
#include <bewego/geodesic_flow/attractors.h>
#include <bewego/motion/differentiable_kinematics.h>
#include <bewego/numerical_optimization/bewopt_robot.h>
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

// # Attractor
// planner.add_flag("attractor_type", "euclidean")
// planner.add_flag("attractor_constraint", True)
// planner.add_flag("attractor_make_smooth", True)
// planner.add_flag("attractor_transition", .15)
// planner.add_flag("attractor_interval", .10)
// planner.add_flag("attractor_squared_potential", False)

RobotOptimizer::RobotOptimizer(
    uint32_t n,                                   // dim of the c-space
    uint32_t T,                                   // number of cliques
    double dt,                                    // time between configs
    const std::vector<double>& workspace_bounds,  // workspace bounds
    std::shared_ptr<const Robot> robot            // kinematics
    )
    : TrajectoryOptimizer(T, dt, n),
      robot_(robot),
      workspace_dim_(3),
      workspace_bounds_(workspace_bounds),
      end_effector_id_(0),
      attractor_type_("euclidean"),
      attractor_transition_(.05),
      attractor_interval_(.02),
      attractor_value_geodesic_(true),
      attractor_make_smooth_(true),
      clique_collision_constraints_(false),
      geodesic_flow_(DifferentiableMapPtr()) {
  if (T_ < 2) {
    throw std::runtime_error(
        "RobotOptimizer : T should be 2 at least, got -> (" +
        std::to_string(T_) + ")");
  }

  if (workspace_dim_ != 2 && workspace_dim_ != 3) {
    throw std::runtime_error(
        "RobotOptimizer : ws dimension missmatch (should be 2 or 3) ( " +
        std::to_string(workspace_dim_) + " )");
  }

  if (robot_.get() == nullptr) {
    throw std::runtime_error("RobotOptimizer : robot not initialized");
  }

  uint32_t c_space_dim = robot_->keypoint_map(0)->input_dimension();
  if (c_space_dim != n) {
    throw std::runtime_error("RobotOptimizer : cspace dimension missmatch ( " +
                             std::to_string(c_space_dim) + " , " +
                             std::to_string(n) + " )");
  }

  if (workspace_.get() == nullptr) {
    throw std::runtime_error("RobotOptimizer : workspace not initialized");
  }

  cout << " -- create collision constraint" << endl;
  // Initialize smooth collision constraint
  auto collision_checker = std::make_shared<SmoothCollisionPointsConstraint>(
      robot_->GetCollisionPoints(),          // Robot keypoints surfaces
      workspace_->ExtractSurfaceFunctions()  // Workspace obstacles surfaces
  );

  cout << " -- get smooth constraint" << endl;
  smooth_collision_constraint_ = collision_checker->smooth_constraint();
}

std::vector<util::Bounds> RobotOptimizer::DofsBounds() const {
  assert(workspace_bounds_.dim() == workspace_dim_);
  std::vector<util::Bounds> limits(n_);
  for (uint32_t i = 0; i < workspace_dim_; i++) {
    limits[i].lower_ = workspace_bounds_.extent(i, ExtentBox::Min);
    limits[i].upper_ = workspace_bounds_.extent(i, ExtentBox::Max);
  }
  return limits;
}

DifferentiableMapPtr RobotOptimizer::GetDistanceActivation() const {
  // First interate through all the key points in the environment
  // Add a constraint per keypoint on the robot.
  // TODO have a different model for the robot (with capsules or ellipsoids)
  VectorOfMaps sdfs;
  for (auto& sdf : workspace_->ExtractSurfaceFunctions()) {
    auto r = robot_->keypoint_radius(0);
    sdfs.push_back(sdf - r);
  }
  auto stack = std::make_shared<CombinedOutputMap>(sdfs);
  auto smooth_min = std::make_shared<NegLogSumExp>(sdfs.size(), robot_gamma_);
  return LogisticActivation(ComposedWith(smooth_min, stack), robot_k_);
}

void RobotOptimizer::AddGeodesicFlowTerm(double scalar) {
  // Add term to agree with flow
  if (scalar <= 0.) return;
  assert(geodesic_flow_.get() != nullptr);

  cout << "workspace_dim_ : " << workspace_dim_ << endl;

  auto position = std::make_shared<Position>(workspace_dim_);
  auto velocity = std::make_shared<Velocity>(workspace_dim_);
  auto normalize = std::make_shared<NormalizeMap>(workspace_dim_);

  // Normalized the flow and velocity
  auto n_vel = ComposedWith(normalize, velocity);
  auto n_flow = ComposedWith(normalize, ComposedWith(geodesic_flow_, position));
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

  cout << "T : " << function_network_->T() << endl;
  for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
    // Calculate PosVel of FK(q)
    auto positions = std::make_shared<MultiEvalMap>(
        robot_->keypoint_map(i), function_network_->nb_clique_elements());
    auto posvel = std::make_shared<FiniteDifferencesPosVel>(n_, dt_);
    auto f = ComposedWith(angle, ComposedWith(posvel, positions));
    for (uint32_t t = 5; t <= T_ - 2; ++t) {
      function_network_->RegisterFunctionForClique(t, dt_ * scalar * f);
    }
  }
}

void RobotOptimizer::AddGeodesicTerms(double scalar, double alpha) {
  // Penalize all dimensions equally, because the scaling of the dimensions
  // should already be handled by the workspace geometry map, itself.

  /**

    TODO Debug this.

   */

  if (scalar <= 0.) return;
  auto workspace_potential = std::make_shared<WorkspacePotentalPrimitive>(
      workspace_->objects(), alpha);
  auto phi = workspace_potential->WorkspaceGeometryMap();
  auto ws_dim = robot_->keypoint_map(0)->output_dimension();
  auto ws_map_dim = phi->output_dimension();

  Eigen::VectorXd regularizers = Eigen::VectorXd::Ones(phi->output_dimension());

  for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
    for (uint32_t t = 0; t <= T_ - 2; ++t) {
      // Penalization of velocity in
      // workspace geometry map (phi):
      //
      //        1/2 | d phi(q)|^2 = 1/2 | J_phi dx |^2

      auto potential = ComposedWith(
          std::make_shared<SquaredNorm>(ws_map_dim),
          ComposedWith(std::make_shared<RangeSubspaceMap>(
                           2 * ws_map_dim, range(ws_map_dim, 2 * ws_map_dim)),
                       std::make_shared<PosVelDifferentiableMap>(phi)));

      // Calculate PosVel of FK(q)
      auto fk = std::make_shared<MultiEvalMap>(robot_->keypoint_map(i), 2);
      auto positions = ComposedWith(fk, function_network_->LeftOfCliqueMap());
      auto posvel = ComposedWith(
          std::make_shared<FiniteDifferencesPosVel>(ws_dim, dt_), positions);

      // cout << "posvel_from_pos->input_dimension() : "
      //      << posvel->input_dimension() << endl;
      // cout << "positions->output_dimension() : "
      //      << positions->output_dimension() << endl;

      // auto f = ComposedWith(potential, pos_vel);

      // TEST
      auto f = ComposedWith(potential, posvel);

      // auto phi = std::make_shared<SquaredNorm>(3 * n_);

      // Scale and register
      function_network_->RegisterFunctionForClique(t, dt_ * scalar * f);
    }
  }
}

void RobotOptimizer::AddKeyPointsSurfaceConstraints(double scalar) {
  if (scalar <= 0.) return;
  uint32_t dim = function_network_->input_dimension();
  if (clique_collision_constraints_) {
    // Set up surface constraints for key points.
    for (uint32_t t = 0; t < T_; ++t) {
      auto network = std::make_shared<FunctionNetwork>(dim, n_);
      auto center_clique = function_network_->CenterOfCliqueMap();
      auto phi = ComposedWith(smooth_sdf_[t], center_clique);
      network->RegisterFunctionForClique(t, dt_ * scalar * phi);
      g_constraints_.push_back(network);
    }
  } else {
    // Set up surface constraints for key points.
    for (uint32_t t = 0; t < T_; ++t) {
      for (uint32_t i = 0; i < robot_->keypoints().size(); i++) {
        auto p = robot_->keypoint_map(i);
        auto r = robot_->keypoint_radius(i);
        assert(p->input_dimension() == n_);
        for (auto& surface_function : workspace_->ExtractSurfaceFunctions()) {
          auto network = std::make_shared<FunctionNetwork>(dim, n_);
          auto phi = ComposedWith(scalar * (surface_function - r), p);
          network->RegisterTimeCenteredPositionFunction(t, phi);
          g_constraints_.push_back(network);
        }
      }
    }
  }
}

/*
void RobotOptimizer::AddKeyPointsSurfaceConstraints(double margin,
                                                       double scalar) {
  if (scalar <= 0.) return;
  if (workspace_objects_.empty()) {
    cerr << "WARNING: no obstacles are in the workspace" << endl;
    return;
  }
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
*/

void RobotOptimizer::AddPosturalTerms(double scalar) {
  if (scalar == 0.) return;
  auto default_q_potential = std::make_shared<SquaredNorm>(q_default_);
  for (uint32_t t = 0; t < T_; t++) {
    function_network_->RegisterTimeCenteredPositionFunction(
        t, dt_ * scalar * default_q_potential);
  }
}

void RobotOptimizer::AddTerminalEndeffectorPotentialTerms(
    const Eigen::VectorXd& x_goal, double scalar) {
  if (x_goal.size() != 3) {
    throw std::runtime_error(
        "RobotOptimizer (AddTerminalEndeffectorPotentialTerms)"
        ": x_goal.size() != 3");
  }
  auto square_dist = std::make_shared<SquaredNorm>(x_goal);
  auto end_effector_pos = robot_->keypoint_map(end_effector_id_);
  auto terminal_potential = ComposedWith(square_dist, end_effector_pos);
  auto center_clique = function_network_->CenterOfCliqueMap();
  auto phi = ComposedWith(terminal_potential, center_clique);
  function_network_->RegisterFunctionForLastClique(scalar * phi);
}

void RobotOptimizer::AddGoalConstraint(const Eigen::VectorXd& x_goal,
                                       double scalar) {
  if (scalar <= 0) return;
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
      double d_t = attractor_transition_;  // where the transition happens
      double d_i = attractor_interval_;    // how fast is the transition
      if (attractor_value_geodesic_) {
        attractor = geodesic_distance_;
        if (attractor_make_smooth_) {
          attractor = std::make_shared<SmoothAttractor>(
              ComposedWith(std::make_shared<SquareMap>(), geodesic_distance_),
              p_goal, d_t, d_i);
        }
      } else {
        // MAN IF THAT WORKS !!!!
        std::dynamic_pointer_cast<const GeodesicDistance>(geodesic_distance_)
            ->set_value(std::make_shared<NaturalAttractor>(
                workspace_->WorkspaceGeometryMap(), p_goal, true));
        if (attractor_make_smooth_) {
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
      // attractor =
      //     ComposedWith(std::make_shared<ScalarSquaredDifference>(0),
      //     attractor);
    }
  }

  // Register
  attractor = scalar * attractor;

  if (kinematic_map) {
    attractor = ComposedWith(attractor, kinematic_map);
  }

  uint32_t dim = function_network_->input_dimension();
  auto network = std::make_shared<FunctionNetwork>(dim, n_);
  network->RegisterTimeCenteredPositionFunction(T_, attractor);
  h_constraints_.push_back(network);
}
