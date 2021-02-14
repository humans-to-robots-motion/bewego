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
#pragma once

#include <bewego/differentiable_map.h>
#include <bewego/trajectory.h>

#include <iostream>

namespace bewego {

class MotionOptimizationFactory {
  MotionOptimizationFactory(uint32_t T, uint32_t t) {}

  /** Apply the following euqation to all cliques:

              | d^n/dt^n x_t |^2

      where n is either 
        1: velocity
        2: accleration
   */
  void AddSmoothnessTerms(uint32_t deriv_order = 2, double scalar) {
    if (deriv_order == 1) {
      auto derivative = std::make_shared<Compose>(
          std::make_shared<SquaredNormVelocity>(config_space_dim_, dt_),
          function_network_->LeftOfCliqueMap());
      function_network_->RegisterFunctionForAllCliques(
          std::make_shared<Scale>(derivative, scalar));
    } else if (deriv_order == 2) {
      auto derivative =
          std::make_shared<SquaredNormAcceleration>(config_space_dim_, dt_);
      function_network_.RegisterFunctionForAllCliques(
          std::make_shared<Scale>(derivative, scalar));
      else {
        std::cerr << "deriv_order (" << deriv_order << ") not suported"
                  << std::endl;
      }
    }
  }

  /** Apply the following euqation to all cliques:

              c(x_t) | d/dt x_t |

          The resulting Riemanian metric is isometric. TODO see paper.
          Introduced in CHOMP, Ratliff et al. 2009. */
  void AddIsometricPotentialToAllCliques(DifferentiableMapPtr potential,
                                         double scalar) {
    auto cost = std::make_shared<Compose>(
        potential, function_network_->CenterOfCliqueMap());
    auto squared_norm_vel = std::make_shared<Compose>(
        std::make_shared<SquaredNormVelocity>(config_space_dim_, dt_),
        function_network_->RightOfCliqueMap());
    function_network_->RegisterFunctionForAllCliques(std::make_shared<Scale>(
        std::make_shared<ProductMap>(cost, squared_norm_vel), scalar));
  }

 protected:
  std::shared_ptr<CliquesFunctionNetwork> function_network_;
  double dt_;
  double config_space_dim_;
};

}  // namespace bewego