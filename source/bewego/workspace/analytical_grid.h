/*
 * Copyright (c) 2016
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
 *                                               Jim Mainprice Wed 4 Feb 2016
 */

// authors: Jim Mainprice, Nathan Ratliff

#pragma once

#include <bewego/derivatives/differentiable_map.h>
#include <bewego/workspace/voxel_grid.h>

#include <Eigen/Core>

namespace bewego {

/**
 * \brief An analytical grid is a discrete voxel grid interfaced through a
 * locally weighted regressor. The resulting object behaves as though it were a
 * continuous function despite its underlying discrete representation.
 */
class AnalyticalGrid : public VoxelGrid<double> {
 public:
  AnalyticalGrid() {}

  /**
   * \brief cell_size: the size of the side of each cubic voxel.
   * env_box: contains, in this order,
   * (min_x, max_x, min_y, max_y, min_z, max_z).
   * mahalanobis_metric: The mahalanobis metric used for calculating grid
   * point proximities.
   * neighborhood_threshold: Positive values give a desired neighborhood
   * threshold in units of meters. In that case, the mahalanobis_metric is
   * used only up to a proportionality. Negative values use the
   * mahalanobis_metric at face value.
   */
  AnalyticalGrid(double cell_size, const std::vector<double>& env_box,
                 const Eigen::MatrixXd& mahalanobis_metric,
                 double neighborhood_threshold, bool corner = false);

  AnalyticalGrid(double cell_size, const std::vector<double>& env_box,
                 double neighborhood_threshold, bool corner = false);

  AnalyticalGrid(double cell_size, const std::vector<double>& env_box,
                 bool corner = false);

  virtual ~AnalyticalGrid() {}

  const std::vector<Eigen::Vector3i>& neighbors() const { return neighbors_; }

  // \brief ! Set the LWR distance metric and computed neighboors
  void SetMahalanobisDistanceMetric(const Eigen::MatrixXd& mahalanobis_metric,
                                    double neighborhood_threshold = -1,
                                    double weight_threshold = 1e-10);

  // \brief ! Get potential by regression
  double CalculatePotential(const Eigen::Vector3d& point) const;

 protected:
  bool ExtractRegressorData(const Eigen::Vector3d& p, Eigen::MatrixXd* X,
                            Eigen::VectorXd* Y) const;
  void ComputeNeighbors(double distance_threshold);

  Eigen::MatrixXd mahalanobis_metric_;
  std::vector<Eigen::Vector3i> neighbors_;
};

/**
 * \brief Lightweight wrapper around the Analytical grid to implement the
 * NDimZerothOrderFunction interface. Upon construction can decide whether the
 * to use the finite-differenced Hessian or to replace it with the identity
 * matrix.
 */
class RegressedVoxelGrid : public DifferentiableMap {
 public:
  RegressedVoxelGrid() { InitDataStructures(); }

  // \brief Keeps a shared pointer to the provided analytical grid.
  RegressedVoxelGrid(std::shared_ptr<const AnalyticalGrid> analytical_grid,
                     bool use_identity_hessian = false)
      : analytical_grid_(analytical_grid),
        use_identity_hessian_(use_identity_hessian) {
    InitDataStructures();
  }

  // \brief Copies the provided analytical grid into this object.
  RegressedVoxelGrid(const AnalyticalGrid& analytical_grid,
                     bool use_identity_hessian = false)
      : analytical_grid_(new AnalyticalGrid(analytical_grid)),
        use_identity_hessian_(use_identity_hessian) {
    InitDataStructures();
  }

  // \brief Keeps a shared pointer to the provided analytical grid.
  void Initalize(std::shared_ptr<const AnalyticalGrid> analytical_grid,
                 bool use_identity_hessian = false) {
    analytical_grid_ = analytical_grid;
    set_use_identity_hessian(use_identity_hessian);
  }

  // \brief Copies the provided analytical grid into this object.
  void Initalize(const AnalyticalGrid& analytical_grid,
                 bool use_identity_hessian = false) {
    analytical_grid_.reset(new AnalyticalGrid(analytical_grid));
    set_use_identity_hessian(use_identity_hessian);
  }

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    y_[0] = analytical_grid_->CalculatePotential(x);
    return y_;
  }

  // When evaluating the Hessian, replace with the identity matrix.
  virtual Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const {
    if (use_identity_hessian_) {
      return identity_;
    } else {
      return FiniteDifferenceHessian(*this, x);
    }
  }

  /* \brief The input dimension will always be 3 because the underlying voxel
   * grid is in 3-space.
   */
  uint32_t input_dimension() const { return 3; }
  uint32_t output_dimension() const { return 1; }

  void set_analytical_grid(std::shared_ptr<const AnalyticalGrid> v) {
    analytical_grid_ = v;
  }

  void set_use_identity_hessian(bool v) { use_identity_hessian_ = v; }

 protected:
  void InitDataStructures() {
    PreAllocate();
    identity_ = Eigen::MatrixXd::Identity(input_dimension(), input_dimension());
  }

  bool use_identity_hessian_;
  std::shared_ptr<const AnalyticalGrid> analytical_grid_;
  Eigen::MatrixXd identity_;
};

/**
 * \brief Returns the 1/2 the square of the values. Useful when the analytical
 * grid represents a distance field. The negative gradient then is the vector
 * pointing to the closest point: -grad( 1/2 d^2 ) = -d grad(d). grad(d) is a
 * unit vector pointing toward the closest obstacle, so scaling it by the
 * distance gives us the desired vector.)
 */
class SquaredRegressedVoxelGrid : public DifferentiableMap {
 public:
  SquaredRegressedVoxelGrid() { PreAllocate(); }

  // \brief Keeps a shared pointer to the provided analytical grid.
  SquaredRegressedVoxelGrid(
      std::shared_ptr<const AnalyticalGrid> analytical_grid)
      : analytical_grid_(analytical_grid) {
    PreAllocate();
  }

  // \brief Copies the provided analytical grid into this object.
  SquaredRegressedVoxelGrid(const AnalyticalGrid& analytical_grid)
      : analytical_grid_(new AnalyticalGrid(analytical_grid)) {
    PreAllocate();
  }

  //\brief Keeps a shared pointer to the provided analytical grid.
  void Initalize(std::shared_ptr<const AnalyticalGrid> analytical_grid) {
    analytical_grid_ = analytical_grid;
  }

  //\brief Copies the provided analytical grid into this object.
  void Initalize(const AnalyticalGrid& analytical_grid) {
    analytical_grid_.reset(new AnalyticalGrid(analytical_grid));
  }

  virtual Eigen::VectorXd Forward(const Eigen::VectorXd& x) const {
    double v = analytical_grid_->CalculatePotential(x);
    y_[0] = .5 * v * v;
    return y_;
  }

  Eigen::Vector3d CalculateNegativeGradient(const Eigen::Vector3d& x) const {
    return -Gradient(x);
  }

  /**
   * \brief The input dimension will always be 3 because the underlying voxel
   * grid is in 3-space.
   */
  uint32_t input_dimension() const { return 3; }
  uint32_t output_dimension() const { return 1; }

 protected:
  std::shared_ptr<const AnalyticalGrid> analytical_grid_;
};

inline double MahalanobisSquareDistance(const Eigen::VectorXd& x1,
                                        const Eigen::VectorXd& x2,
                                        const Eigen::MatrixXd& D) {
  Eigen::VectorXd diff = x1 - x2;
  return diff.transpose() * D * diff;
}

inline double MahalanobisDistance(const Eigen::VectorXd& x1,
                                  const Eigen::VectorXd& x2,
                                  const Eigen::MatrixXd& D) {
  return sqrt(MahalanobisSquareDistance(x1, x2, D));
}

inline double DistanceAtWeightThreshold(double weight_threshold) {
  return sqrt(-2 * log(weight_threshold));
}

/**
 * \brief Rescale the Mahalanobis metric so that the weight at the specified
 * distance is the given threshold. D is both an input parameter specifying the
 * Mahalanobis metric and an output parameter storing the scaled matrix.
 * It's assumed that distance_threshold is in units given by the Mahalanobis
 * metric.
 * Algebra: exp{ -s d^2/2} = w, solve for s (d is the distance threshold, w is
 * the
 * corresponding weight threshold). Solution: s = -2/d^2 log(w)
 */
inline void RescaleMahalanobisMetric(double distance_threshold,
                                     double corresponding_weight_threshold,
                                     Eigen::MatrixXd* D) {
  double d_squared = distance_threshold * distance_threshold;
  *D *= (-2. / d_squared) * log(corresponding_weight_threshold);
}

}  // namespace bewego
