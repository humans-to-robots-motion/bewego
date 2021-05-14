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

#include <bewego/util/interpolation.h>
#include <bewego/workspace/analytical_grid.h>

#include <Eigen/LU>  // for matrix inverse
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <set>

namespace bewego {

AnalyticalGrid::AnalyticalGrid(double cell_size,
                               const std::vector<double>& env_box,
                               const Eigen::MatrixXd& mahalanobis_metric,
                               double neighborhood_threshold,
                               bool corner)
    : VoxelGrid<double>(
          env_box[1] - env_box[0],  // size_x
          env_box[3] - env_box[2],  // size_y
          env_box[5] - env_box[4],  // size_z
          cell_size,
          corner ? env_box[0] + cell_size * 0.5 : env_box[0],  // origin_x
          corner ? env_box[2] + cell_size * 0.5 : env_box[2],  // origin_y
          corner ? env_box[4] + cell_size * 0.5 : env_box[4],  // origin_z
          double(0.0)) {
  SetMahalanobisDistanceMetric(mahalanobis_metric, neighborhood_threshold);
}

AnalyticalGrid::AnalyticalGrid(double cell_size,
                               const std::vector<double>& env_box,
                               double neighborhood_threshold,
                               bool corner)
    : VoxelGrid<double>(
          env_box[1] - env_box[0],  // size_x
          env_box[3] - env_box[2],  // size_y
          env_box[5] - env_box[4],  // size_z
          cell_size,
          corner ? env_box[0] + cell_size * 0.5 : env_box[0],  // origin_x
          corner ? env_box[2] + cell_size * 0.5 : env_box[2],  // origin_y
          corner ? env_box[4] + cell_size * 0.5 : env_box[4],  // origin_z
          double(0.0)) {
  SetMahalanobisDistanceMetric(Eigen::MatrixXd::Identity(3, 3),
                               neighborhood_threshold);
}

AnalyticalGrid::AnalyticalGrid(double cell_size,
                               const std::vector<double>& env_box,
                               bool corner)
    : VoxelGrid<double>(
          env_box[1] - env_box[0],  // size_x
          env_box[3] - env_box[2],  // size_y
          env_box[5] - env_box[4],  // size_z
          cell_size,
          corner ? env_box[0] + cell_size * 0.5 : env_box[0],  // origin_x
          corner ? env_box[2] + cell_size * 0.5 : env_box[2],  // origin_y
          corner ? env_box[4] + cell_size * 0.5 : env_box[4],  // origin_z
          double(0.0)) {}

void AnalyticalGrid::SetMahalanobisDistanceMetric(
    const Eigen::MatrixXd& mahalanobis_metric, double neighborhood_threshold,
    double weight_threshold) {
  mahalanobis_metric_ = mahalanobis_metric;
  RescaleMahalanobisMetric(neighborhood_threshold, weight_threshold,
                           &mahalanobis_metric_);

  std::cout << __PRETTY_FUNCTION__ << std::endl;
  ComputeNeighbors(DistanceAtWeightThreshold(weight_threshold));
  std::cout << "number of neighboors : " << neighbors_.size() << std::endl;
}

void AnalyticalGrid::ComputeNeighbors(double distance_threshold) {
  neighbors_.clear();
  std::vector<Eigen::Vector3i> new_neighbours;
  std::vector<Eigen::Vector3i> neighbours;

  new_neighbours.push_back(Eigen::Vector3i::Zero());
  neighbors_.push_back(Eigen::Vector3i::Zero());
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(3);

  // Not efficient, but this is done as a preprocessing step.
  while (!new_neighbours.empty()) {
    neighbours = new_neighbours;
    new_neighbours.clear();
    // cout << "neighbors_.size() : " << neighbors_.size() << endl;

    for (int j = 0; j < neighbours.size(); j++) {
      for (int i = 0; i < 26; i++) {  // All neighbours of neighbours
        Eigen::Vector3d offset;
        offset[0] = (i / 1) % 3 - 1 + neighbours[j][0];
        offset[1] = (i / 3) % 3 - 1 + neighbours[j][1];
        offset[2] = (i / 9) % 3 - 1 + neighbours[j][2];

        Eigen::Vector3d coord;
        coord[0] = resolution_ * double(offset[0]);
        coord[1] = resolution_ * double(offset[1]);
        coord[2] = resolution_ * double(offset[2]);

        double distance = MahalanobisDistance(coord, zero, mahalanobis_metric_);
        if (distance <= distance_threshold) {
          Eigen::Vector3i offset_tmp;
          offset_tmp << int(offset[0]), int(offset[1]), int(offset[2]);

          if (std::find(neighbors_.begin(), neighbors_.end(), offset_tmp) ==
              neighbors_.end()) {
            new_neighbours.push_back(offset_tmp);
            neighbors_.push_back(offset_tmp);
          }
        }
      }
    }
  }
}

bool AnalyticalGrid::ExtractRegressorData(const Eigen::Vector3d& p,
                                          Eigen::MatrixXd* X,
                                          Eigen::VectorXd* Y) const {
  assert(X != nullptr);
  assert(Y != nullptr);

  // Don't care if the resulting coordinate is out of bounds. Points outside the
  // grid may still have some relevant grid point neighbors, using those ensures
  // that the regression degrades smoothly away from the voxel grid.
  Eigen::Vector3i coord;
  worldToGrid(p, coord);

  X->resize(neighbors_.size(), 3);  // Data points
  Y->resize(neighbors_.size());     // Function valid

  int j = 0;  // Valid data points id

  for (size_t i = 0; i < neighbors_.size(); i++) {
    Eigen::Vector3i coord_neigh = coord + neighbors_[i];

    if (isCellValid(coord_neigh)) {
      Eigen::Vector3d center;
      gridToWorld(coord_neigh, center);
      X->row(j) = center;
      (*Y)(j) = getCell(coord_neigh);

      j++;  // Increment number of data points
    }
  }

  X->conservativeResize(j, 3);
  Y->conservativeResize(j);

  return true;
}

double AnalyticalGrid::CalculatePotential(const Eigen::Vector3d& p) const {
  Eigen::MatrixXd X;
  Eigen::VectorXd Y;
  ExtractRegressorData(p, &X, &Y);
  return CalculateLocallyWeightedRegression(p, X, Y, mahalanobis_metric_, 1e-6);
}

}  // namespace bewego
