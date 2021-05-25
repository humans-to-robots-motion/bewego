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
#include <bewego/motion/cost_terms.h>

using namespace bewego;

//------------------------------------------------------------------------------
// PosVelDifferentiableMap implementation
//------------------------------------------------------------------------------

PosVelDifferentiableMap::PosVelDifferentiableMap(DifferentiableMapPtr phi)
    : phi_(phi), n_(phi_->input_dimension()) {
  PreAllocate();
  J_.setZero();
  type_ = "PosVelDifferentiableMap";
}

Eigen::VectorXd PosVelDifferentiableMap::Forward(
    const Eigen::VectorXd& x) const {
  CheckInputDimension(x);

  const Eigen::VectorXd& q = x.head(n_);   // position
  const Eigen::VectorXd& qd = x.tail(n_);  // velocity

  y_.head(phi_->output_dimension()) = (*phi_)(q);              // phi_q
  y_.tail(phi_->output_dimension()) = phi_->Jacobian(q) * qd;  // phi_qd

  return y_;
}

Eigen::MatrixXd PosVelDifferentiableMap::Jacobian(
    const Eigen::VectorXd& x) const {
  const Eigen::VectorXd& q = x.head(n_);   // position
  const Eigen::VectorXd& qd = x.tail(n_);  // velocity

  Eigen::MatrixXd J_phi = phi_->Jacobian(q);  // J
  Eigen::MatrixXd Jd_phi;                     // J dot (finite difference)
  const double dt = 1e-4;
  Jd_phi = phi_->Jacobian(q + 0.5 * dt * qd);
  Jd_phi -= phi_->Jacobian(q - 0.5 * dt * qd);
  Jd_phi /= dt;

  uint32_t m_phi = phi_->output_dimension();
  uint32_t n_phi = phi_->input_dimension();
  J_.topLeftCorner(m_phi, n_phi) = J_phi;
  J_.bottomRightCorner(m_phi, n_phi) = J_phi;
  J_.bottomLeftCorner(m_phi, n_phi) = Jd_phi;
  return J_;
}
