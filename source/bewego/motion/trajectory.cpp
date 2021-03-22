/*
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
 *                                                             Thu 11 Feb 2021
 */
// author: Jim Mainprice, mainprice@gmail.com

#include <bewego/motion/trajectory.h>
#include <bewego/util/misc.h>
#include <bewego/util/multi_variate_gaussian.h>

using std::cout;
using std::endl;

namespace bewego {

DifferentiableMapPtr TrajectoryConstraintNetwork(uint32_t T, uint32_t n,
                                                 DifferentiableMapPtr g,
                                                 double gamma) {
  uint32_t dimension = (T + 1) * n;
  std::vector<DifferentiableMapPtr> configurations_maps(T + 1);
  for (uint32_t t = 0; t < T + 1; t++) {
    std::vector<uint32_t> q_indices;
    for (uint32_t i = 0; i < n; i++) {
      q_indices.push_back(t * n + i);
    }
    configurations_maps[t] = ComposedWith(
        g, std::make_shared<RangeSubspaceMap>(dimension, q_indices));
  }
  auto stack = std::make_shared<CombinedOutputMap>(configurations_maps);
  auto smooth_min = std::make_shared<NegLogSumExp>(T + 1, gamma);
  return ComposedWith(smooth_min, stack);
  // return std::make_shared<Min>(configurations_maps);
}

std::shared_ptr<Trajectory> InitializeZeroTrajectory(
    const Eigen::VectorXd& q_init, uint32_t T) {
  uint32_t n = q_init.size();
  auto trajectory = std::make_shared<Trajectory>(n, T);
  for (uint32_t t = 0; t <= T + 1; ++t) {
    trajectory->Configuration(t) = q_init;
  }
  return trajectory;
}

Trajectory GetTrajectory(const std::vector<Eigen::VectorXd>& line) {
  Trajectory trajectory(line.front().size(), line.size() - 1);
  for (uint32_t t = 0; t <= trajectory.T(); t++) {
    trajectory.Configuration(t) = line[t];
  }
  return trajectory;
}

Trajectory GetLinearInterpolation(const Eigen::VectorXd& q_init,
                                  const Eigen::VectorXd& q_goal, uint32_t T) {
  Trajectory trajectory(q_init.size(), T);
  for (uint32_t t = 0; t <= T + 1; t++) {
    double alpha = double(t) / double(T);
    if (alpha > 1) {
      alpha = 1;
    }
    trajectory.Configuration(t) = (1. - alpha) * q_init + alpha * q_goal;
  }
  return trajectory;
}

Trajectory Resample(const Trajectory& trajectory, uint32_t T) {
  Trajectory resampled_trajectory(trajectory.Configuration(0), T);
  ContinuousTrajectory continous_trajectory(trajectory);
  for (uint32_t t = 0; t <= T; t++) {
    double s = double(t) / double(T);
    resampled_trajectory.Configuration(t) =
        continous_trajectory.ConfigurationAtParameter(s);
  }
  return resampled_trajectory;
}

//! Samples trajectories with 0 mean
/**
std::vector<Trajectory> SampleTrajectoriesZeroMean(
    uint32_t T, double dt, uint32_t n, uint32_t nb_samples, double std_dev) {
  // Get the precision matrix using the hessian of the control costs
  Eigen::MatrixXd precision = GetControlCostPrecisionMatrix(T, dt, n);

  // Inverse of the precison
  Eigen::MatrixXd covariance = precision.inverse();

  // Get gaussian sampler
  auto sampler = std::make_shared<MultivariateGaussian>(
      Eigen::VectorXd::Zero(precision.rows()), covariance);

  // Sample all trajectories, set the active segment
  // set the standard deviation
  std::vector<Trajectory> samples(nb_samples);
  for (uint32_t i = 0; i < nb_samples; i++) {
    Trajectory sample(n, T);
    Eigen::VectorXd x = sample.ActiveSegment();
    sampler->sample(x);
    x *= std_dev;
    sample.ActiveSegment() = x;
    samples[i] = sample;
  }
  return samples;
}
*/

bool AreTrajectoriesEqual(const Trajectory& traj1, const Trajectory& traj2,
                          double error) {
  if (traj1.n() != traj2.n()) {
    return false;
  }
  if (traj1.T() != traj2.T()) {
    return false;
  }
  if (error == 0) {
    for (uint32_t t = 0; t <= traj1.T() + 1; t++) {
      if (traj1.Configuration(t) != traj2.Configuration(t)) {
        return false;
      }
    }
  } else {
    assert(error == 0);
    for (uint32_t t = 0; t <= traj1.T() + 1; t++) {
      double delta = (traj1.Configuration(t) - traj2.Configuration(t)).norm();
      if (delta > error) {
        cout << "t = " << t << endl;
        cout << "delta = " << delta << endl;
        return false;
      }
    }
  }
  return true;
}

bool IsTrajectoryNullMotion(const Trajectory& trajectory, double error) {
  auto q_init = trajectory.Configuration(0);
  if (error == 0) {
    for (uint32_t t = 1; t <= trajectory.T() + 1; t++) {
      if (trajectory.Configuration(t) != q_init) {
        return false;
      }
    }
  } else {
    assert(error == 0);
    for (uint32_t t = 1; t <= trajectory.T() + 1; t++) {
      if ((trajectory.Configuration(t) - q_init).norm() > error) {
        return false;
      }
    }
  }
  return true;
}

bool DoesTrajectoryCollide(
    const Trajectory& trajectory,
    std::shared_ptr<const DifferentiableMap> collision_check, double margin) {
  // Check only inner part of the trajectory.
  bool collide = false;
  for (int t = 1; t < trajectory.T(); t++) {
    if (collision_check->ForwardFunc(trajectory.Configuration(t)) < margin) {
      collide = true;
      cout << "collide @ t = " << t << endl;
    }
  }
  return collide;
}

Trajectory InitializeFromMatrix(const Eigen::MatrixXd& Xi) {
  Trajectory trajectory(Xi.cols(), Xi.rows() - 2);
  for (uint32_t t = 0; t <= trajectory.T() + 1; t++) {
    trajectory.Configuration(t) = Xi.row(t);
  }

  return trajectory;
}

}  // namespace bewego
