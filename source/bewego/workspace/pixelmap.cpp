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

#include <bewego/util/misc.h>
#include <bewego/workspace/extent.h>
#include <bewego/workspace/pixelmap.h>

// Standard and other includes
#include <iomanip>
#include <iostream>
#include <sstream>

using std::cout;
using std::endl;

namespace bewego {

// Returns the matrix of values
Eigen::MatrixXd PixelMap::GetMatrix(uint32_t buffer) const {
  uint32_t nb_cell_x = num_cells_x_ - 2 * buffer;
  uint32_t nb_cell_y = num_cells_y_ - 2 * buffer;
  Eigen::MatrixXd mat(nb_cell_y, nb_cell_x);
  Eigen::Vector2i loc;
  for (loc.x() = buffer; loc.x() < (num_cells_x_ - buffer); loc.x()++) {
    for (loc.y() = buffer; loc.y() < (num_cells_y_ - buffer); loc.y()++) {
      mat(loc.y() - buffer, loc.x() - buffer) = GetCell(loc);
    }
  }
  return mat;
}

// Get data from matrix
void PixelMap::InitializeFromMatrix(const Eigen::MatrixXd& mat,
                                    uint32_t buffer) {
  assert(num_cells_y_ == mat.rows() + 2 * buffer);
  assert(num_cells_x_ == mat.cols() + 2 * buffer);
  Eigen::Vector2i loc;
  for (loc.x() = 0; loc.x() < num_cells_x_; loc.x()++) {
    for (loc.y() = 0; loc.y() < num_cells_y_; loc.y()++) {
      if (loc.y() < buffer || loc.x() < buffer ||
          loc.y() >= (num_cells_y_ - buffer) ||
          loc.x() >= (num_cells_x_ - buffer)) {
        GetCell(loc) = default_value_;
      } else {
        GetCell(loc) = mat(loc.y() - buffer, loc.x() - buffer);
      }
    }
  }
}

// TODO write that more efficiently
// This is n^2 and can be done in n
void PixelMap::SetCrown(double v, uint32_t id) {
  assert(num_cells_x_ > 2 * (id + 1));
  assert(num_cells_y_ > 2 * (id + 1));
  uint32_t dim = std::max(num_cells_x_, num_cells_y_);
  Eigen::Vector2i loc;
  for (uint32_t i = id; i < dim; i++) {
    if (i < num_cells_x_ - id) {
      loc.x() = i;
      loc.y() = id;
      GetCell(loc) = v;
      loc.y() = num_cells_y_ - id - 1;
      GetCell(loc) = v;
    }
    if (i < num_cells_y_ - id) {
      loc.x() = id;
      loc.y() = i;
      GetCell(loc) = v;
      loc.x() = num_cells_x_ - id - 1;
      GetCell(loc) = v;
    }
  }
}

// TODO test
void PixelMap::SetBorderCrown(double v, uint32_t buffer) {
  assert(num_cells_y_ > 2 * buffer);
  assert(num_cells_x_ > 2 * buffer);
  for (uint32_t i = 1; i <= buffer; i++) {
    SetCrown(v, i - 1);
  }
}

bool Check2DGridsEqual(const PixelMap& grid1, const PixelMap& grid2,
                       double tolerance) {
  if ((grid1.num_cells_x() != grid2.num_cells_x()) ||
      (grid1.num_cells_y() != grid2.num_cells_y())) {
    return false;
  }
  Eigen::Vector2i loc;
  for (loc.x() = 0; loc.x() < grid1.num_cells_x(); loc.x()++) {
    for (loc.y() = 0; loc.y() < grid1.num_cells_y(); loc.y()++) {
      double value_1 = grid1.GetCell(loc);
      double value_2 = grid2.GetCell(loc);
      if (tolerance > 0) {
        if (std::fabs(value_1 - value_2) > tolerance) {
          return false;
        }
      } else {
        if (value_1 != value_2) {
          return false;
        }
      }
    }
  }
  return true;
}

Eigen::MatrixXd EvaluateFunction(const DifferentiableMap& f,
                                 const PixelMap& grid) {
  Eigen::MatrixXd values(grid.num_cells_x(), grid.num_cells_y());
  Eigen::Vector2i loc;
  Eigen::Vector2d world;
  int dim_x = util::uint_to_int(grid.num_cells_x());
  int dim_y = util::uint_to_int(grid.num_cells_y());
  for (loc.x() = 0; loc.x() < dim_x; loc.x()++) {
    for (loc.y() = 0; loc.y() < dim_y; loc.y()++) {
      grid.GridToWorld(loc, world);
      values(loc.y(), loc.x()) = f.ForwardFunc(world);
    }
  }

  return values;
}

void InitializeDataFromFunction(const DifferentiableMap& f, PixelMap& grid,
                                double height) {
  Eigen::Vector2i loc;
  Eigen::Vector2d world(0, 0);
  Eigen::Vector3d world_3d(0, 0, height);
  uint32_t dim = f.input_dimension();
  int dim_x = util::uint_to_int(grid.num_cells_x());
  int dim_y = util::uint_to_int(grid.num_cells_y());
  for (loc.x() = 0; loc.x() < dim_x; loc.x()++) {
    for (loc.y() = 0; loc.y() < dim_y; loc.y()++) {
      grid.GridToWorld(loc, world);
      if (dim == 2) {
        grid.GetCell(loc) = f.ForwardFunc(world);
      } else {
        world_3d.segment(0, 2) = world;
        grid.GetCell(loc) = f.ForwardFunc(world_3d);
      }
    }
  }
}

std::shared_ptr<PixelMap> Discretize2DFunction(
    double resolution, const ExtentBox& extends,
    std::shared_ptr<const DifferentiableMap> f) {
  assert(2 == f->input_dimension());
  auto grid = std::make_shared<PixelMap>(resolution, extends);
  InitializeDataFromFunction(*f, *grid);
  return grid;
}

bool SavePixelMapToFile(const PixelMap& grid, const std::string& filename,
                        uint32_t buffer) {
  Eigen::MatrixXd values(grid.GetMatrix(buffer));
  return util::SaveMatrixToCsvFile(filename, values);
}

}  // namespace bewego
