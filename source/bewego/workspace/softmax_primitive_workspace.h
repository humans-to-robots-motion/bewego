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

#pragma once

#include <bewego/derivatives/differentiable_map.h>
#include <bewego/workspace/workspace.h>

namespace bewego {

/**  f(x) = rho_scaling * exp(-alpha * SDF(x))  */
class ObstaclePotential : public CombinationOperator {
 public:
  ObstaclePotential() { type_ = "ObstaclePotential"; }
  ObstaclePotential(DifferentiableMapPtr sdf, double alpha, double rho_scaling);
  virtual ~ObstaclePotential() {}

  Eigen::VectorXd Forward(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const;
  Eigen::MatrixXd Hessian(const Eigen::VectorXd& x) const;

  // Returns the dimension of the domain (input) space.
  virtual uint32_t input_dimension() const { return ambient_space_dim_; }
  virtual uint32_t output_dimension() const { return 1; }

  virtual VectorOfMaps nested_operators() const {
    return VectorOfMaps({signed_distance_field_});
  }

  // Returns a pointer to the sdf
  DifferentiableMapPtr signed_distance_field() const {
    return signed_distance_field_;
  }

 private:
  uint32_t ambient_space_dim_;
  DifferentiableMapPtr signed_distance_field_;
  double alpha_;
  double rho_scaling_;
};

/// Multipotential System Map
class MultiObstaclePotentialMap : public DifferentiableMap {
 public:
  // Set the object coordinate system regressed functions
  // ρ : potential, higher when proximity to the object is high
  MultiObstaclePotentialMap(const VectorOfWorkpaceObjects& potentials,
                            double alpha, double scalar = 1.);

  // Evaluates x = phi(q) and returns x in the return parameter. Derived classes
  // should validate that x isn't null.
  Eigen::VectorXd Forward(const Eigen::VectorXd& q) const;

  // Calculates the Jacobian of the map at q (J = d/dq phi(q)) and returns the
  // resulting matrix at J. Derived classes should validate that J isn't null.
  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& q) const;

  virtual uint32_t input_dimension() const { return ambient_space_dim_; }
  virtual uint32_t output_dimension() const { return obstacle_space_dim_; }

 protected:
  void PreAllocateJacobian();
  uint32_t ambient_space_dim_;
  uint32_t obstacle_space_dim_;
  std::vector<std::shared_ptr<const ObstaclePotential>> rho_;
};

//\ Class that represents a workspace composed of a signed distance field
class WorkspacePotentalPrimitive : public Workspace {
 public:
  WorkspacePotentalPrimitive(const VectorOfWorkpaceObjects& objects,
                             double alpha, double scalar = 1.)
      : Workspace(objects) {
    Initialize(objects, alpha, scalar);
  }

  virtual ~WorkspacePotentalPrimitive();

  void Initialize(const VectorOfWorkpaceObjects& objects, double alpha,
                  double scalar);

 private:
  DifferentiableMapPtr workspace_geometry_map_;
};

}  // namespace bewego
