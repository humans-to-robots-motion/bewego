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

#include <bewego/numerical_optimization/bewopt_planar.h>

#include <Eigen/Core>

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace bewego::util;
using std::cerr;
using std::cout;
using std::endl;

// int32        ff_ipopt_maximum_iters, 200
// string       ff_ipopt_hessian_approximation, "limited-memory"
// double       ff_ipopt_constr_viol_tol, 1e-7
// double       ff_ipopt_acceptable_tol, 1e-6
// double       ff_ipopt_tol, 1e-3
// bool         ff_ipopt_with_bounds, true

PlanarOptimizer::PlanarOptimizer(uint32_t T, double dt,
                                 const std::vector<double>& workspace_bounds)
    : TrajectoryOptimizer(T, dt, 2) {
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

void PlanarOptimizer::AddGoalManifoldConstraint(const Eigen::VectorXd& q_goal,
                                                double radius, double scalar) {
  assert(function_network_.get() != nullptr);
  assert(n_ == 2);

  uint32_t dim = function_network_->input_dimension();
  auto network = std::make_shared<FunctionNetwork>(dim, n_);

  cout << "Create constraint manifold at : " << q_goal.transpose() << endl;

  // Create clique constraint function phi
  auto d_goal = std::make_shared<SphereDistance>(q_goal, radius);
  auto d_sq_goal = ComposedWith(std::make_shared<SquareMap>(), d_goal);
  auto soft_dist = std::make_shared<SoftDist>(d_sq_goal);
  auto phi = ComposedWith(soft_dist, network->CenterOfCliqueMap());

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

void PlanarOptimizer::AddWayPointManifoldConstraint(
    const Eigen::VectorXd& q_waypoint, uint32_t t, double radius,
    double scalar) {
  assert(function_network_.get() != nullptr);
  assert(n_ == 2);

  uint32_t dim = function_network_->input_dimension();
  auto network = std::make_shared<FunctionNetwork>(dim, n_);

  // Create clique constraint function phi
  auto d_waypoint = std::make_shared<SphereDistance>(q_waypoint, radius);
  auto d_sq_waypoint = ComposedWith(std::make_shared<SquareMap>(), d_waypoint);
  auto soft_dist = std::make_shared<SoftDist>(d_sq_waypoint);
  auto phi = ComposedWith(soft_dist, network->LeftMostOfCliqueMap());

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

void PlanarOptimizer::AddSmoothKeyPointsSurfaceConstraints(double scalar) {
  if (workspace_objects_.empty()) {
    cerr << "WARNING: no obstacles are in the workspace" << endl;
    return;
  }
  assert(function_network_.get() != nullptr);
  assert(n_ == 2);

  // we use a vector of smooth distance to efficiently use cache
  uint32_t dim = function_network_->input_dimension();
  for (uint32_t t = 0; t < T_; t++) {
    auto network = std::make_shared<FunctionNetwork>(dim, n_);
    auto center_clique = function_network_->CenterOfCliqueMap();
    auto phi = ComposedWith(smooth_sdf_[t], center_clique);
    network->RegisterFunctionForClique(t, dt_ * scalar * phi);
    g_constraints_.push_back(network);
  }
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
