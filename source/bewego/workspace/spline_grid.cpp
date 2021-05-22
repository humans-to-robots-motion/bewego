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

#include <bewego/util/misc.h>
#include <bewego/workspace/spline_grid.h>

using namespace bewego;

//-----------------------------------------------------------------------------
// AnalyticPixelMapSpline implementation.
//-----------------------------------------------------------------------------

void AnalyticPixelMapSpline::InitializeSplines() {
  offset_ = 0.5 * resolution() * Eigen::Vector2d::Ones() + origin_minus();
  interpolator_ = std::make_shared<BiCubicGridInterpolator>(
      data_, resolution(), num_cells_x(), num_cells_y());
}

//! Get potential by regression
double AnalyticPixelMapSpline::CalculateSplineValue(
    const Eigen::Vector2d& point) const {
  Eigen::Vector2i pos;
  if (!WorldToGrid(point, pos)) {
    return default_value_;
  }
  return interpolator_->Evaluate(point - offset_);
}

//! Get potential by regression
Eigen::Vector2d AnalyticPixelMapSpline::CalculateSplineGradient(
    const Eigen::Vector2d& point) const {
  assert(g->size() == 2);
  double sval, sgrad[2];
  // Trick the compiler by copying the pointer
  // this library is written in C and does not declare its arguments as const
  Eigen::Vector2i pos;
  if (!WorldToGrid(point, pos)) {
    return Eigen::Vector2d::Zero();
  }
  return interpolator_->Gradient(point - offset_);
}

// Get the spline value and gradient
Eigen::Matrix2d AnalyticPixelMapSpline::CalculateSplineGradientHessian(
    const Eigen::Vector2d& point) const {
  assert(g->size() == 2);
  assert(H->rows() == 2);
  assert(H->cols() == 2);
  double sval, sgrad[2], shess[4];
  // Trick the compiler by copying the pointer
  // this library is written in C and does not declare its arguments as const
  Eigen::Vector2i pos;
  if (!WorldToGrid(point, pos)) {
    return Eigen::Matrix2d::Zero();
  }
  return Eigen::Matrix2d::Identity();
}

//-----------------------------------------------------------------------------
// RegressedPixelGridSpline implementation.
//-----------------------------------------------------------------------------

Eigen::VectorXd RegressedPixelGridSpline::Forward(
    const Eigen::VectorXd& x) const {
  if (outside_function_) {
    Eigen::Vector2i pos;
    if (!analytical_grid_->WorldToGrid(Eigen::Vector2d(x), pos)) {
      return (*outside_function_)(x);
    }
  }
  y_[0] = analytical_grid_->CalculateSplineValue(Eigen::Vector2d(x));
  return y_;
}

Eigen::MatrixXd RegressedPixelGridSpline::Jacobian(
    const Eigen::VectorXd& x) const {
  Eigen::Vector2d x_2d(x);
  if (outside_function_) {
    Eigen::Vector2i pos;
    if (!analytical_grid_->WorldToGrid(x_2d, pos)) {
      return outside_function_->Jacobian(x);
    }
  }
  return analytical_grid_->CalculateSplineGradient(x_2d).transpose();
  ;
}

Eigen::MatrixXd RegressedPixelGridSpline::Hessian(
    const Eigen::VectorXd& x) const {
  Eigen::Vector2d x_2d(x);
  if (outside_function_) {
    Eigen::Vector2i pos;
    if (!analytical_grid_->WorldToGrid(x_2d, pos)) {
      return outside_function_->Hessian(x);
    }
  }
  if (use_identity_hessian_) {
    // H_.setIdentity();
    // Not identity but Gauss-Netwon approximation
    // *H = (*g) * (*g).transpose();
    // cout << "H : " << *H << endl;
    // cout << "g : " << (*g).transpose() << endl;
    return H_;
  } else {
    // double value = Evaluate(x, g);

    return analytical_grid_->CalculateSplineGradientHessian(x_2d);

    /*
    //  Finite difference hessian
    lula::differential_geometry::NDimDiffFunction::
        GetFiniteDifferenceHessianFromGradients(x, &H_approx);

    double max_H_diff = ((*H) - H_approx).cwiseAbs().maxCoeff();
    cout << "max_H_diff : " << max_H_diff << endl;
    cout << " -- H : " << (*H) << endl;
    cout << " -- H_approx : " << H_approx << endl;
    cout << " -- H_approx : " << g_2d.transpose() << endl;
    */
    // CHECK_LE(max_H_diff, 1e-3);
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

std::shared_ptr<RegressedPixelGridSpline> bewego::RegressedGridFrom2DFunction(
    double resolution, const extent_t& extends,
    std::shared_ptr<const DifferentiableMap> f) {
  assert(2 == f->input_dimension());
  auto analytic_grid =
      std::make_shared<AnalyticPixelMapSpline>(resolution, extends);
  InitializeDataFromFunction(*f, *analytic_grid);
  analytic_grid->InitializeSplines();
  return std::make_shared<RegressedPixelGridSpline>(analytic_grid);
}

std::shared_ptr<RegressedPixelGridSpline>
bewego::InitializeRegressedPixelGridFromMatrix(double resolution,
                                               const Eigen::MatrixXd& values,
                                               uint32_t buffer) {
  double x_width = (values.cols() + 2 * buffer) * resolution;
  double y_width = (values.rows() + 2 * buffer) * resolution;
  extent_t extends;
  extends.extents_.resize(4);
  extends.extents_[1] = x_width / 2.;
  extends.extents_[3] = y_width / 2.;
  extends.extents_[0] = -extends.x_max();
  extends.extents_[2] = -extends.y_max();
  auto analytic_grid =
      std::make_shared<AnalyticPixelMapSpline>(resolution, extends);
  // This value should not be too high to
  // not make the problem numerically unstable
  analytic_grid->set_default_value(values.maxCoeff());
  analytic_grid->InitializeFromMatrix(values, buffer);
  analytic_grid->InitializeSplines();
  return std::make_shared<RegressedPixelGridSpline>(analytic_grid);
}

std::shared_ptr<RegressedPixelGridSpline>
bewego::InitializeRegressedPixelGridFromExpMatrix(double resolution,
                                                  const Eigen::MatrixXd& values,
                                                  bool exponentiated,
                                                  uint32_t buffer) {
  assert(is_nan(values) == false);
  assert(values.rows() == 100);
  assert(values.cols() == 100);
  if (exponentiated) {
    Eigen::MatrixXd values_tmp = values;
    util::ExponentiateMatrix(values_tmp);
    // cout << "MAX COEEF : " << values_tmp.maxCoeff() << endl;
    return InitializeRegressedPixelGridFromMatrix(resolution, values_tmp,
                                                  buffer);
  }
  return InitializeRegressedPixelGridFromMatrix(resolution, values, buffer);
}

std::shared_ptr<RegressedPixelGridSpline>
bewego::LoadRegressedPixelGridFromFile(double resolution,
                                       const std::string& filename,
                                       bool exponentiated, uint32_t buffer) {
  std::shared_ptr<RegressedPixelGridSpline> regressed_grid;
  Eigen::MatrixXd values(util::ReadMatrixFromCsvFile(filename));
  if (values.rows() == 0 || values.cols() == 0) {
    return regressed_grid;
  }
  if (exponentiated) {
    util::ExponentiateMatrix(values);
  }
  return InitializeRegressedPixelGridFromMatrix(resolution, values, buffer);
}

/* TODO

void AnalyticalGridSpline::Recompute() {
  CHECK_NOTNULL(splines_);
  data_einslpine_ = &data_[0];
  recompute_UBspline_3d_d(splines_, data_einslpine_);
}

void AnalyticalGridSpline::Initialize() {
  Ugrid x_grid, y_grid, z_grid;

  x_grid.start = origin_[DIM_X];
  x_grid.end = x_grid.start + num_cells_[DIM_X] * resolution_;
  x_grid.num = num_cells_[DIM_X];

  y_grid.start = origin_[DIM_Y];
  y_grid.end = y_grid.start + num_cells_[DIM_Y] * resolution_;
  y_grid.num = num_cells_[DIM_Y];

  z_grid.start = origin_[DIM_Z];
  z_grid.end = z_grid.start + num_cells_[DIM_Z] * resolution_;
  z_grid.num = num_cells_[DIM_Z];

  BCtype_d xBC, yBC, zBC;
  xBC.lCode = xBC.rCode = PERIODIC;
  yBC.lCode = yBC.rCode = PERIODIC;
  zBC.lCode = zBC.rCode = PERIODIC;

  //  int nb_points = num_cells_[DIM_X] * num_cells_[DIM_Y] * num_cells_[DIM_Z];
  //  data_einslpine_ = new double[nb_points];
  //  for (uint32_t x = 0; x < num_cells_[DIM_X]; ++x) {
  //    for (uint32_t y = 0; y < num_cells_[DIM_Y]; ++y) {
  //      for (uint32_t z = 0; z < num_cells_[DIM_Z]; ++z) {
  //        data_einslpine_[x * stride1_ + y * stride2_ + z] = getCell(x, y, z);
  //      }
  //    }
  //  }

  data_einslpine_ = &data_[0];

  cout << "create_UBspline_3d_d" << endl;
  splines_ = create_UBspline_3d_d(x_grid, y_grid, z_grid, xBC, yBC, zBC,
                                  data_einslpine_);
}

//! Get potential by regression
double AnalyticalGridSpline::CalculateSplineGradient(
    const Eigen::Vector3d& point, Eigen::Vector3d* g) const {
  CHECK_EQ(g->size(), 3);
  double sval, sgrad[3];
  // Trick the compiler by copying the pointer
  // this library is written in C and does not declare its arguments as const
  Eigen::Vector3i pos;
  if (!worldToGrid(point, pos)) {
    (*g) = Eigen::Vector3d::Zero();
    // cout << "Return default value : " << point.transpose() << endl;
    return default_value_;
  }
  UBspline_3d_d* splines_copy = splines_;
  eval_UBspline_3d_d_vg(splines_copy, point.x(), point.y(), point.z(), &sval,
                        sgrad);
  g->x() = sgrad[0];
  g->y() = sgrad[1];
  g->z() = sgrad[2];
  return sval;
}

// Get the spline value and gradient
double AnalyticalGridSpline::CalculateSplineGradientHessian(
    const Eigen::Vector3d& point, Eigen::Vector3d* g,
    Eigen::Matrix3d* H) const {
  CHECK_EQ(g->size(), 3);
  CHECK_EQ(H->rows(), 3);
  CHECK_EQ(H->cols(), 3);
  double sval, sgrad[3], shess[9];
  // Trick the compiler by copying the pointer
  // this library is written in C and does not declare its arguments as const
  Eigen::Vector3i pos;
  if (!worldToGrid(point, pos)) {
    (*g) = Eigen::Vector3d::Zero();
    (*H) = Eigen::Matrix3d::Zero();
    return default_value_;
  }
  UBspline_3d_d* splines_copy = splines_;
  eval_UBspline_3d_d_vgh(splines_copy, point.x(), point.y(), point.z(), &sval,
                         sgrad, shess);
  g->x() = sgrad[0];
  g->y() = sgrad[1];
  g->z() = sgrad[2];
  // TODO: figure out convention
  (*H) << shess[0], shess[3], shess[6], shess[1], shess[4], shess[7], shess[2],
      shess[5], shess[8];
  return sval;
}

//! Get potential by regression
double AnalyticalGridSpline::CalculateSplineValue(
    const Eigen::Vector3d& point) const {
  Eigen::Vector3i pos;
  if (!worldToGrid(point, pos)) {
    return default_value_;
  }
  double sval;
  // Trick the compiler by copying the pointer
  // this library is written in C and does not declare its arguments as const
  UBspline_3d_d* splines_copy = splines_;
  eval_UBspline_3d_d(splines_copy, point.x(), point.y(), point.z(), &sval);
  return sval;
}

//! Get the triliearly interpolated potential
//!
//! TODO TEST AND FINISH ....
double AnalyticalGridSpline::TrilinearInterpolation(
    const Eigen::Vector3d& point) const {
  Eigen::Vector3i grid;
  worldToGrid(point, grid);

  Eigen::Vector3d p_min;
  gridToWorld(grid + Eigen::Vector3i(-1, -1, -1), p_min);
  // TODO add is cell valid

  // Eigen::Vector3d p_max;
  // gridTWorld( grid + Eigen::Vector3i(1,1,1), p_max);
  // TODO add is cell valid

  double interpolation_length = 2. * resolution_;

  double xd = (point.x() - p_min.x()) / interpolation_length;
  double yd = (point.y() - p_min.y()) / interpolation_length;
  double zd = (point.z() - p_min.z()) / interpolation_length;

  double c00 = getCell(grid + Eigen::Vector3i(-1, -1, -1)) * (1 - xd) +
               getCell(grid + Eigen::Vector3i(1, -1, -1)) * xd;
  double c10 = getCell(grid + Eigen::Vector3i(-1, 1, -1)) * (1 - xd) +
               getCell(grid + Eigen::Vector3i(1, 1, -1)) * xd;
  double c01 = getCell(grid + Eigen::Vector3i(-1, -1, 1)) * (1 - xd) +
               getCell(grid + Eigen::Vector3i(1, -1, 1)) * xd;
  double c11 = getCell(grid + Eigen::Vector3i(-1, 1, 1)) * (1 - xd) +
               getCell(grid + Eigen::Vector3i(1, 1, 1)) * xd;

  double c0 = c00 * (1 - yd) + c10 * yd;
  double c1 = c01 * (1 - yd) + c11 * yd;

  return c0 * (1 - zd) + c1 * zd;
}

//------------------------------------------------------------------------------
// RegressedVoxelGridSpline implementation.
//------------------------------------------------------------------------------

double RegressedVoxelGridSpline::Evaluate(const Eigen::VectorXd& x) const {
  std::shared_ptr<const AnalyticalGridSpline> grid =
      std::static_pointer_cast<const AnalyticalGridSpline>(analytical_grid_);
  if (outside_function_) {
    Eigen::Vector3i pos;
    if (!grid->worldToGrid(Eigen::Vector3d(x), pos)) {
      return (*outside_function_)(x);
    }
  }
  return grid->CalculateSplineValue(Eigen::Vector3d(x));
}

double RegressedVoxelGridSpline::Evaluate(const Eigen::VectorXd& x,
                                          Eigen::VectorXd* g) const {
  std::shared_ptr<const AnalyticalGridSpline> grid =
      std::static_pointer_cast<const AnalyticalGridSpline>(analytical_grid_);
  if (outside_function_) {
    Eigen::Vector3i pos;
    if (!analytical_grid_->worldToGrid(Eigen::Vector3d(x), pos)) {
      return outside_function_->Evaluate(x, g);
    }
  }
  Eigen::Vector3d g_3d;
  double value = grid->CalculateSplineGradient(Eigen::Vector3d(x), &g_3d);
  *g = g_3d;
  return value;
}

// When evaluating the Hessian, replace with the identity matrix.
double RegressedVoxelGridSpline::Evaluate(const Eigen::VectorXd& x,
                                          Eigen::VectorXd* g,
                                          Eigen::MatrixXd* H) const {
  if (use_identity_hessian_) {
    CHECK_NOTNULL(g);
    CHECK_NOTNULL(H);
    double value = Evaluate(x, g);
    // *H = (*g) * (*g).transpose();
    *H = Eigen::MatrixXd::Identity(input_dimension(), input_dimension());
    // cout << "H : " << *H << endl;
    // cout << "g : " << (*g).transpose() << endl;
    return value;
  } else {
    double value = Evaluate(x, g);
    Eigen::Vector3d g_3d;
    Eigen::Matrix3d H_3d;

    std::shared_ptr<const AnalyticalGridSpline> grid =
        std::static_pointer_cast<const AnalyticalGridSpline>(analytical_grid_);

    value =
        grid->CalculateSplineGradientHessian(Eigen::Vector3d(x), &g_3d, &H_3d);
    *H = H_3d;

    //  Finite difference hessian
    //    Eigen::MatrixXd H_approx = Eigen::MatrixXd::Zero(3, 3);

    //        GetFiniteDifferenceHessianFromGradients(x, &H_approx);

    // double max_H_diff = ((*H) - H_approx).cwiseAbs().maxCoeff();
    // cout << "max_H_diff : " << max_H_diff << endl;
    // cout << " -- H : " << (*H) << endl;
    // cout << " -- H_approx : " << H_approx << endl;
    // cout << " -- H_g : " << (*g) * (*g).transpose() << endl;
    // cout << " -- g : " << (*g).transpose() << endl;
    // CHECK_LE(max_H_diff, 1e-3);
    return value;
  }
}

//-----------------------------------------------------------------------------
// RegressedVoxelGridOffset implementation.
//-----------------------------------------------------------------------------

double RegressedVoxelGridOffset::Evaluate(const Eigen::VectorXd& x) const {
  Eigen::Vector3d x_local = object_pose_inv_ * Eigen::Vector3d(x);
  double dist = x_local.norm();
  Eigen::Vector3i coord;
  // cout << "evaluate constraints : " << x_local.transpose() << endl;
  if (dist > 2.0 || !analytical_grid_->worldToGrid(x_local, coord)) {
    return dist;  // in units of meters (TODO: set a paramter of the class)
  }

  return RegressedVoxelGridSpline::Evaluate(x_local);
}

double RegressedVoxelGridOffset::Evaluate(const Eigen::VectorXd& x,
                                          Eigen::VectorXd* g,
                                          Eigen::MatrixXd* H) const {
  Eigen::Vector3d x_local = object_pose_inv_ * Eigen::Vector3d(x);
  double dist = x_local.norm();
  Eigen::Vector3i coord;
  // cout << "evaluate constraints : " << x_local.transpose() << endl;
  if (dist > 2.0 || !analytical_grid_->worldToGrid(x_local, coord)) {
    // cout << "out side of sdf" << endl;
    return dist;  // in units of meters (TODO: set a paramter of the class)
  }
  double potential = RegressedVoxelGridSpline::Evaluate(x_local, g, H);
  Eigen::Vector3d g_3d(*g);  // 3d gradient
  *g = object_pose_inv_.rotation().transpose() * g_3d;
  *H = g_3d * g_3d.transpose();
  return potential;
}

*/