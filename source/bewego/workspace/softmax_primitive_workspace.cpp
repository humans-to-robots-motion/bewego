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
 *                                               Jim Mainprice Fri 16 Apr 2021
 */

#include <bewego/derivatives/differentiable_map.h>
#include <bewego/workspace/softmax_primitive_workspace.h>

#include <memory>

using namespace std;
using namespace bewego;

namespace bewego {

//------------------------------------------------------------------------------
// ObstaclePotential
//------------------------------------------------------------------------------

ObstaclePotential::ObstaclePotential(
    std::shared_ptr<const DifferentiableMap> sdf, double alpha,
    double rho_scaling) {
  signed_distance_field_ = sdf;
  ambient_space_dim_ = sdf->input_dimension();
  alpha_ = alpha;
  rho_scaling_ = rho_scaling;
  type_ = "ObstaclePotential";
}

Eigen::VectorXd ObstaclePotential::Forward(const Eigen::VectorXd& x) const {
  assert(x.size() == ambient_space_dim_);
  double sd = signed_distance_field_->Forward(x)[0];
  double rho = rho_scaling_ * exp(-alpha_ * sd);
  return Eigen::VectorXd::Constant(1, rho);
}

Eigen::MatrixXd ObstaclePotential::Jacobian(const Eigen::VectorXd& x) const {
  assert(x.size() == ambient_space_dim_);
  double rho = Forward(x)[0];
  return -alpha_ * rho * signed_distance_field_->Jacobian(x);
}

Eigen::MatrixXd ObstaclePotential::Hessian(const Eigen::VectorXd& x) const {
  assert(x.size() == ambient_space_dim_);
  double rho = Forward(x)[0];
  auto J_sdf = signed_distance_field_->Jacobian(x);
  auto H_sdf = signed_distance_field_->Hessian(x);
  return rho * (alpha_ * alpha_ * J_sdf.transpose() * J_sdf - alpha_ * H_sdf);
}

//------------------------------------------------------------------------------
// MultiObstaclePotentialMap
//------------------------------------------------------------------------------

MultiObstaclePotentialMap::MultiObstaclePotentialMap(
    const std::vector<std::shared_ptr<const WorkspaceObject>>& objects,
    double alpha, double scalar) {
  for (auto& o : objects) {
    auto signed_distance_field = o->ConstraintFunction();
    rho_.push_back(std::make_shared<ObstaclePotential>(signed_distance_field,
                                                       alpha, scalar));
  }
  ambient_space_dim_ = rho_.back()->input_dimension();
  obstacle_space_dim_ = rho_.size() + ambient_space_dim_;
  PreAllocate();
}

Eigen::VectorXd MultiObstaclePotentialMap::Forward(
    const Eigen::VectorXd& p) const {
  assert(p.rows() == ambient_space_dim_);
  for (uint32_t i = 0; i < rho_.size(); i++) {
    y_[i] = rho_[i]->ForwardFunc(p);
  }
  y_.tail(ambient_space_dim_) = p;
  return y_;
}

void MultiObstaclePotentialMap::PreAllocateJacobian() {
  // Preallocate Jacobian constent parts.
  const uint32_t& m = obstacle_space_dim_;
  const uint32_t& n = ambient_space_dim_;
  J_.block(m - n, 0, n, n) = Eigen::MatrixXd::Identity(n, n);
}

Eigen::MatrixXd MultiObstaclePotentialMap::Jacobian(
    const Eigen::VectorXd& p) const {
  assert(p.rows() == ambient_space_dim_);
  for (uint32_t i = 0; i < rho_.size(); i++) {
    J_.row(i) = rho_[i]->Gradient(p).transpose();
  }
  return J_;
}

//------------------------------------------------------------------------------
// WorkspacePotentalPrimitive
//------------------------------------------------------------------------------

WorkspacePotentalPrimitive::~WorkspacePotentalPrimitive() {}

void WorkspacePotentalPrimitive::Initialize(
    const VectorOfWorkpaceObjects& objects, double alpha, double scalar) {
  workspace_geometry_map_ =
      make_shared<MultiObstaclePotentialMap>(objects, alpha, scalar);
}

}  // namespace bewego
