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
#pragma once

// #include <bewego/workspace/analytical_grid.h>
#include <bewego/freeflyers.h>
#include <bewego/motion/motion_optimization.h>
#include <bewego/motion/trajectory_function_network.h>
#include <bewego/numerical_optimization/optimizer.h>
#include <bewego/numerical_optimization/trajectory_optimization.h>
#include <bewego/stats_monitor.h>
#include <bewego/trajectory_optimization_freeflyer.h>
#include <bewego/workspace/analytical_workspace.h>

namespace bewego {

/**
 * Box that can bin inializeed from 6 floating point numbers and then quierried
 * easily for extent. By default initializes to a unit cube.
 */
struct Box {
  enum Type { Min, Max };
  Box() : data(std::vector<double>{0, 1, 0, 1, 0, 1}) {}
  Box(double x_min, double x_max, double y_min, double y_max, double z_min,
      double z_max)
      : data(std::vector<double>{x_min, x_max, y_min, y_max, z_min, z_max}) {}
  double extent(uint32_t i, Type b) const {
    assert(i <= 3);
    return data[b == Min ? 2 * i : 2 * i + 1];
  }
  friend ostream& operator<<(ostream& os, const Box& b);
  std::vector<double> data;
};

ostream& operator<<(ostream& os, const Box& b) {
  for (const auto& x : b.data) os << x << " ";
  return os;
}

class FreeflyerOptimzer : public TrajectoryOptimizer {
 public:
  FreeflyerOptimzer(uint32_t n,  // size of the configuration space
                    uint32_t T,  // number of cliques
                    double dt,   // time between cliques
                    Box workspace_bounds, std::shared_ptr<Workspace> workspace,
                    std::shared_ptr<Freeflyer> robot);

  /** @brief set_parameters_from_flags */
  void set_parameters_from_flags();

  /** @brief Adds a geodesic flow object to the optimizer */
  void set_geodesic_flow(std::shared_ptr<const DifferentiableMap> v) {
    geodesic_flow_ = v;
  }

  /** @brief Adds a geodesic flow object to the optimizer */
  void set_geodesic_distance(std::shared_ptr<const DifferentiableMap> v) {
    geodesic_distance_ = v;
  }

  /** @brief Adds a default configuration */
  void set_q_default(const Eigen::VectorXd& v) { q_default_ = v; }

  /** @brief Sets the end enffector of the freeflyer, this which what part of
   * the geometry should be used for setting up a goal attractor */
  void set_end_effector(uint32_t i) { end_effector_id_ = i; }

  /** @brief return bounds constraints */
  std::vector<BoundConstraint> GetJointLimits() const;

  /** @brief return bounds for dofs along the trajectory */
  std::vector<Bounds> GetDofBounds() const;

  void AddGeodesicFlowTerm() const;
  void AddGeodesicTerm() const;
  void AddInternalAddKeyPointBarriers() const;
  void AddKeyPointsSurfaceConstraints() const;
  void AddJointLimitConstraints() const;
  void AddGoalConstraint(const Eigen::VectorXd& x_goal) const;
  void AddPosturalTerms() const;

  /** @brief Optimize a given trajectory */
  OptimizeResult Optimize(
      const Eigen::VectorXd& initial_traj,          // entire trajectory
      const Eigen::VectorXd& x_goal,                // goal configuration
      const std::map<std::string, double>& options  // optimizer options
  ) const;

 protected:
  typedef CliquesFunctionNetwork FunctionNetwork;
  typedef std::shared_ptr<const FunctionNetwork> FunctionNetworkPtr;
  typedef std::shared_ptr<const DifferentiableMap> ElementaryFunction;

  std::shared_ptr<const ConstrainedOptimizer> SetupIpoptOptimizer(
      const Eigen::VectorXd& q_init) const;

  FunctionNetwork ObjectiveNetwork(const Eigen::VectorXd& x_goal) const;
  FunctionNetwork InequalityConstraints() const;
  FunctionNetwork EqualityConstraints(const Eigen::VectorXd& x_goal) const;

  // Get ditance to obstacle
  ElementaryFunction GetDistanceActivation() const;

  // ipopt constraints
  FunctionNetwork GetEmptyFunctionNetwork() const;
  std::vector<FunctionNetwork> GetKeyPointsSurfaceConstraints() const;

  bool verbose_;
  uint32_t workspace_dim_;  // Dimensionality of the workspace
  uint32_t n_;              // Dimensionality of the configuration space
  uint32_t T_;              // Number of active cliques
  double dt_;               // time interval between cliques

  // Bounds of the workspace
  Box workspace_bounds_;

  // Workspace
  std::shared_ptr<Workspace> workspace_;
  std::shared_ptr<const DifferentiableMap> smooth_collision_constraint_;

  // GeodesicFlow
  std::shared_ptr<const DifferentiableMap> geodesic_flow_;
  std::shared_ptr<const DifferentiableMap> geodesic_distance_;

  // Constraints networks
  std::vector<DifferentiableMapPtr>
      g_constraints_unstructured_;                 // inequalities
  std::vector<FunctionNetworkPtr> g_constraints_;  // inequalities
  std::vector<FunctionNetworkPtr> h_constraints_;  // equalities

  // Robot
  std::shared_ptr<Freeflyer> robot_;

  // End-effector
  uint32_t end_effector_id_;

  // Default posture
  Eigen::VectorXd q_default_;

  // Scalars
  double scalar_cspace_vel_;
  double scalar_cspace_acc_;
  double scalar_flow_;
  double scalar_geodesic_;
  double scalar_goal_constraint_;
  double scalar_obstacle_barrier_;
  double scalar_joint_limits_;
  double scalar_surface_constraint_;
  double scalar_posture_;
  std::string attractor_type_;

  // Parameters
  double freeflyer_gamma_;
  double freeflyer_k_;

  // Logging
  bool visualize_inner_loop_;
  bool monitor_inner_statistics_;
  // mutable std::shared_ptr<FreeflyerOptimizationVisualizer> visualizer_;
  // mutable std::shared_ptr<StatsMonitor> stats_monitor_;
};

}  // namespace bewego
