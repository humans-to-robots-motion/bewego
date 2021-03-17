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
 *                                               Jim Mainprice Wed 10 Mar 2021
 */

#pragma once

#include <bewego/derivatives/differentiable_map.h>
#include <bewego/util/misc.h>
#include <bewego/workspace/extent.h>

#include <Eigen/Core>

namespace bewego {

// Squared PixelMap centered at [0, 0]
// Uses the Gaussians to come up with a binary representation
class PixelMap {
 public:
  PixelMap(double resolution, const extent_t& extends,
           bool origin_center_cell = false) {
    Initialize(resolution, extends.x_min(), extends.x_max(), extends.y_min(),
               extends.y_max(), origin_center_cell);
  }

  PixelMap(double resolution, double x_min, double x_max, double y_min,
           double y_max, bool origin_center_cell = false)
      : resolution_(resolution) {
    Initialize(resolution, x_min, x_max, y_min, y_max, origin_center_cell);
  }

  extent_t GetExtents() const {
    return extent_t(
        origin_minus_.x(), origin_minus_.x() + num_cells_x_ * resolution_,
        origin_minus_.y(), origin_minus_.y() + num_cells_y_ * resolution_);
  }

  bool WorldToGrid(const Eigen::Vector2d& world, Eigen::Vector2i& grid) const {
    Eigen::Vector2d loc = (world - origin_minus_) * oo_resolution_;
    grid.x() = std::floor(loc.x());
    grid.y() = std::floor(loc.y());
    return IsCellValid(grid);
  }

  void GridToWorld(const Eigen::Vector2i& grid, Eigen::Vector2d& world) const {
    Eigen::Vector2d grid_d;
    grid_d << grid.x(), grid.y();
    world = origin_ + resolution_ * grid_d;
  }

  double operator()(const Eigen::Vector2d& world) const {
    Eigen::Vector2i grid;
    if (!WorldToGrid(world, grid)) return default_value_;
    return GetCell(grid);
  }

  bool IsCellValid(const Eigen::Vector2i& grid) const {
    bool x_ok = grid.x() >= 0 && grid.x() < num_cells_x_;
    bool y_ok = grid.y() >= 0 && grid.y() < num_cells_y_;
    return (x_ok && y_ok);
  }

  bool ProjectGridCoordinatesInBounds(Eigen::Vector2i& grid) const {
    bool projected = false;
    if (grid.x() >= num_cells_x_) {
      projected = true, grid.x() = num_cells_x_ - 1;
    }
    if (grid.y() >= num_cells_y_) {
      projected = true, grid.y() = num_cells_y_ - 1;
    }
    if (grid.x() < 0) {
      projected = true, grid.x() = 0;
    }
    if (grid.y() < 0) {
      projected = true, grid.y() = 0;
    }
    return projected;
  }

  double GetCell(const Eigen::Vector2i& coord) const {
    return data_[ref(coord)];  // Const access
  }
  double& GetCell(const Eigen::Vector2i& coord) {
    return data_[ref(coord)];  // Mutable
  }

  // Returns the matrix of values
  Eigen::MatrixXd GetMatrix(uint32_t buffer = 0) const;

  // Get data from matrix
  void InitializeFromMatrix(const Eigen::MatrixXd& mat, uint32_t buffer = 0);

  // Set the crown of the pixelmap to a specific value
  void SetCrown(double v, uint32_t id);

  // Set buffer area around border to value
  void SetBorderCrown(double v, uint32_t buffer = 1);

  // Set the default value.
  void set_default_value(double v) { default_value_ = v; }

  // Get default
  double default_value() const { return default_value_; }

  // Accessors.
  const std::vector<double>& data() const { return data_; }
  const Eigen::Vector2d& origin() const { return origin_; }
  const Eigen::Vector2d& origin_minus() const { return origin_minus_; }
  double resolution() const { return resolution_; }
  uint32_t num_cells_x() const { return num_cells_x_; }
  uint32_t num_cells_y() const { return num_cells_y_; }

 protected:
  void Initialize(double resolution, double x_min, double x_max, double y_min,
                  double y_max, bool origin_center_cell = true) {
    assert(x_max > x_min);
    assert(y_max > y_min);
    resolution_ = resolution;
    num_cells_x_ = util::float_to_uint((x_max - x_min) / resolution_);
    num_cells_y_ = util::float_to_uint((y_max - y_min) / resolution_);
    stride_ = num_cells_y_;  // This matches einspline's format
    // HERE change the origin to be the min and not the center of the cell
    // CHECK whether this matches the einspline format ..
    // TODO write a test script.
    if (origin_center_cell) {
      origin_.x() = x_min;
      origin_.y() = y_min;
      origin_minus_ = origin_ - 0.5 * resolution_ * Eigen::Vector2d::Ones();
    } else {
      origin_minus_.x() = x_min;
      origin_minus_.y() = y_min;
      origin_ = origin_minus_ + 0.5 * resolution_ * Eigen::Vector2d::Ones();
    }
    oo_resolution_ = 1.0 / resolution_;
    default_value_ = std::numeric_limits<double>::infinity();
    data_.resize(num_cells_x_ * num_cells_y_, default_value_);
  }

  // TODO make this match with einspline
  // maybe the stride has to be on x instead of y
  uint32_t ref(const Eigen::Vector2i& coord) const {
    return coord.x() * stride_ + coord.y();
  }

  // Protected members.
  std::vector<double> data_;
  double default_value_;

 private:
  Eigen::Vector2d origin_;
  Eigen::Vector2d origin_minus_;
  double resolution_;
  uint32_t num_cells_x_;
  uint32_t num_cells_y_;
  uint32_t stride_;
  double oo_resolution_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

class BitMap : public PixelMap {
 public:
  BitMap(double threshold, double resolution, const extent_t& exentds)
      : PixelMap(resolution, exentds), threshold_(threshold) {}

  BitMap(double threshold, double resolution, const extent_t& exentds,
         const Eigen::MatrixXd& matrix, bool inverted = false)
      : PixelMap(resolution, exentds), threshold_(threshold) {
    InitializeFromMatrix(matrix);
    InitializeBinaryData(inverted);
    default_value_ = 0.;
  }

  BitMap(double threshold, const PixelMap& grid, bool inverted = true)
      : PixelMap(grid), threshold_(threshold) {
    InitializeBinaryData(inverted);
    default_value_ = 0.;
  }

  // This is typically we construct the BitMap from a signed distance function
  // which is why it is inverted by default.
  void InitializeBinaryData(bool inverted = true) {
    if (inverted) {
      upper_ = 0.;
      lower_ = 1.;
    } else {
      upper_ = 1.;
      lower_ = 0.;
    }
    for (auto& value : data_) value = value > threshold_ ? upper_ : lower_;
  }

  double threshold() const { return threshold_; }

 private:
  double threshold_;
  double upper_;
  double lower_;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Check that the data in the grids is equal
// If no tolerance is passed as argument, will check strong equality
// does not check resolution, only check data
bool Check2DGridsEqual(const PixelMap& grid1, const PixelMap& grid2,
                       double tolerance = -1);

// This function will initalize the data of a PixelMap given a function definied
// everywhere in the world
void InitializeDataFromFunction(const DifferentiableMap& f, PixelMap& grid,
                                double height = 0);

// Same as above but returns matrix
Eigen::MatrixXd EvaluateFunction(const DifferentiableMap& f,
                                 const PixelMap& grid);

// Discretize a 2D function in a pixel map
std::shared_ptr<PixelMap> Discretize2DFunction(
    double resolution, const extent_t& extends,
    std::shared_ptr<const DifferentiableMap> f);

// Save the pixelmap to file
bool SavePixelMapToFile(const PixelMap& grid, const std::string& filename,
                        uint32_t buffer = 0);

}  // namespace bewego
