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

#include <bewego/derivatives/atomic_operators.h>

using namespace bewego;

SecondOrderTaylorApproximation::SecondOrderTaylorApproximation(
    const DifferentiableMap& f, const Eigen::VectorXd& x0)
    : x0_(x0) {
  assert(f.output_dimension() == 1);
  double v = f.ForwardFunc(x0);
  Eigen::VectorXd g = f.Gradient(x0);
  Eigen::MatrixXd H = f.Hessian(x0);
  double c = 0;

  H_ = H;
  a_ = H;
  b_ = g - x0.transpose() * H;
  c = v - g.transpose() * x0 + .5 * x0.transpose() * H * x0;
  c_ = Eigen::VectorXd::Constant(1, c);

  x0_ = x0;
  g0_ = g;
  fx0_ = v;
}