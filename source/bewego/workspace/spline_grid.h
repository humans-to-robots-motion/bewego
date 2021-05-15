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
 *                                               Jim Mainprice Sam 15 May 2021
 */

#pragma once

#include <bewego/workspace/analytical_grid.h>

namespace bewego {

#define DEFAULT_ANALYTICAL std::numeric_limits<double>::max()

class AnalyticalGridSpline : public AnalyticalGrid {
 public:
  AnalyticalGridSpline() : splines_(NULL) {}
  AnalyticalGridSpline(double cell_size, const std::vector<double>& env_box,
                       double neighborhood_threshold, bool corner = false)
      : AnalyticalGrid(cell_size, env_box, neighborhood_threshold, corner),
        splines_(NULL),
        default_value_(DEFAULT_ANALYTICAL) {}
  AnalyticalGridSpline(double cell_size, const std::vector<double>& env_box,
                       bool corner = false)
      : AnalyticalGrid(cell_size, env_box, corner),
        splines_(NULL),
        default_value_(DEFAULT_ANALYTICAL) {}

  //! The analytical grid passed in argument is supposed to be filled
  //! with data, so we call the initialize rootine
  AnalyticalGridSpline(const AnalyticalGridSpline& analytical_grid)
      : AnalyticalGrid(analytical_grid),
        default_value_(analytical_grid.default_value_) {
    Initialize();
  }

  //! The analytical grid passed in argument is supposed to be filled
  //! with data, so we call the initialize rootine
  AnalyticalGridSpline(const AnalyticalGrid& analytical_grid)
      : AnalyticalGrid(analytical_grid), default_value_(DEFAULT_ANALYTICAL) {
    Initialize();
  }
  AnalyticalGridSpline(std::shared_ptr<const AnalyticalGrid> analytical_grid)
      : AnalyticalGrid(*analytical_grid), default_value_(DEFAULT_ANALYTICAL) {
    Initialize();
  }
  AnalyticalGridSpline(std::shared_ptr<const AnalyticalGrid> analytical_grid,
                       double default_value)
      : AnalyticalGrid(*analytical_grid), default_value_(default_value) {
    Initialize();
  }
  virtual ~AnalyticalGridSpline() {
    // delete splines_;
    // delete data_einslpine_;
  }

  //! Get potential by regression
  //  double CalculatePotential(const Eigen::Vector3d& point) const {
  //    return CalculateSplineValue(point);
  //  }

  //! Get potential by regression
  double CalculateSplineValue(const Eigen::Vector3d& point) const;

  //! Get potential and gradient by regression
  double CalculateSplineGradient(const Eigen::Vector3d& point,
                                 Eigen::Vector3d* g) const;

  // Get the spline value and hessian
  double CalculateSplineGradientHessian(const Eigen::Vector3d& point,
                                        Eigen::Vector3d* g,
                                        Eigen::Matrix3d* H) const;

  //! Get the triliearly interpolated potential
  double TrilinearInterpolation(const Eigen::Vector3d& point) const;

  //! Initialize the spline structure
  void Initialize();

  //! ReInitialize the spline structure if the data changed.
  void Recompute();

  //! set default value
  void set_default_value(double v) { default_value_ = v; }

 protected:
  UBspline_3d_d* splines_;
  double* data_einslpine_;
  double default_value_;
};

// Lightweight wrapper around the Analytical grid to implement the
// NDimZerothOrderFunction interface. Upon construction can decide whether the
// to use the finite-differenced Hessian or to replace it with the identity
// matrix.
class RegressedVoxelGridSpline : public RegressedVoxelGrid {
 public:
  RegressedVoxelGridSpline() : RegressedVoxelGrid() {}

  // Normal initialization
  RegressedVoxelGridSpline(std::shared_ptr<const AnalyticalGridSpline> grid)
      : RegressedVoxelGrid(grid, true) {
    std::cout << "Normal intialization" << std::endl;
  }

  // Keeps a shared pointer to the provided analytical grid.
  RegressedVoxelGridSpline(
      std::shared_ptr<const AnalyticalGrid> analytical_grid,
      bool use_identity_hessian)
      : RegressedVoxelGrid(
            std::make_shared<const AnalyticalGridSpline>(analytical_grid),
            use_identity_hessian) {}

  // Copies the provided analytical grid into this object.
  RegressedVoxelGridSpline(const AnalyticalGrid& analytical_grid,
                           bool use_identity_hessian = false)
      : RegressedVoxelGrid(
            std::make_shared<const AnalyticalGridSpline>(analytical_grid),
            use_identity_hessian) {}

  // Copy.
  RegressedVoxelGridSpline(const RegressedVoxelGridSpline& other)
      : RegressedVoxelGrid() {
    // Will replace the analytical grid with a copy for thread safety issues.
    // Will also initialize the spline this operatio is long.
    Initalize(other.analytical_grid_, other.use_identity_hessian_);
  }

  // Cloning
  std::shared_ptr<RegressedVoxelGridSpline> Clone() const {
    return std::make_shared<RegressedVoxelGridSpline>(*this);
  }

  // Keeps a shared pointer to the provided analytical grid.
  void Initalize(std::shared_ptr<const AnalyticalGrid> analytical_grid,
                 bool use_identity_hessian = false) {
    analytical_grid_ =
        std::make_shared<const AnalyticalGridSpline>(analytical_grid);
    set_use_identity_hessian(use_identity_hessian);
  }
  // Copies the provided analytical grid into this object.
  void Initalize(const AnalyticalGrid& analytical_grid,
                 bool use_identity_hessian = false) {
    analytical_grid_.reset(new AnalyticalGridSpline(analytical_grid));
    set_use_identity_hessian(use_identity_hessian);
  }

  // virtual double Evaluate(const Eigen::VectorXd& x) const;
  // virtual double Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g)
  // const; virtual double Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd*
  // g,
  //                         Eigen::MatrixXd* H) const;

  void set_analytical_grid(std::shared_ptr<const AnalyticalGrid> v) {
    analytical_grid_ = std::make_shared<const AnalyticalGridSpline>(v);
  }

  // const accessor to the analytical grid
  std::shared_ptr<const AnalyticalGrid> analytical_grid() const {
    return analytical_grid_;
  }

  void set_outside_function(DifferentiableMapPtr f) { outside_function_ = f; }

 protected:
  DifferentiableMapPtr outside_function_;
};

// Lightweight wrapper around the Analytical grid to implement the
// NDimZerothOrderFunction interface and allows to place
// the Axis aligned grid anyhwere in the world
// this is useful when the grid is centered on an object
class RegressedVoxelGridOffset : public RegressedVoxelGridSpline {
 public:
  RegressedVoxelGridOffset() : RegressedVoxelGridSpline() {}

  // Keeps a shared pointer to the provided analytical grid.
  RegressedVoxelGridOffset(
      const Eigen::Isometry3d& pose,
      std::shared_ptr<const AnalyticalGrid> analytical_grid,
      bool trilinear_interpolation)
      : RegressedVoxelGridSpline(analytical_grid, true),
        trilinear_interpolation_(trilinear_interpolation) {
    SetGridPose(pose);
  }

  // Copies the provided analytical grid into this object.
  RegressedVoxelGridOffset(const Eigen::Isometry3d& pose,
                           const AnalyticalGrid& analytical_grid,
                           bool trilinear_interpolation)
      : RegressedVoxelGridSpline(analytical_grid, true),
        trilinear_interpolation_(trilinear_interpolation) {
    SetGridPose(pose);
  }

  // Keeps a shared pointer to the provided analytical grid.
  void Initalize(const Eigen::Isometry3d& pose,
                 std::shared_ptr<const AnalyticalGrid> analytical_grid) {
    SetGridPose(pose);
    analytical_grid_ =
        std::make_shared<const AnalyticalGridSpline>(analytical_grid);
    set_use_identity_hessian(true);
  }
  // Copies the provided analytical grid into this object.
  void Initalize(const Eigen::Isometry3d& pose,
                 const AnalyticalGrid& analytical_grid) {
    SetGridPose(pose);
    analytical_grid_.reset(new AnalyticalGridSpline(analytical_grid));
    set_use_identity_hessian(true);
  }

  // Set object pose in the world
  void SetGridPose(const Eigen::Isometry3d& pose) {
    object_pose_inv_ = pose.inverse();
  }

  void SetDistanceNeighborhood(double dist) {
    //    analytical_grid_->ComputeNeighbors(dist);
  }

  virtual double Evaluate(const Eigen::VectorXd& x) const;
  virtual double Evaluate(const Eigen::VectorXd& x, Eigen::VectorXd* g,
                          Eigen::MatrixXd* H) const;

  using NDimZerothOrderFunction::Evaluate;

 protected:
  Eigen::Isometry3d object_pose_inv_;
  bool trilinear_interpolation_;
};

}  // namespace bewego