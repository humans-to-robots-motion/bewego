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

#include <bewego/trajectory.h>
#include <bewego/differentiable_map.h>

namespace bewego {

class MotionOptimizationFactory {

    MotionOptimizationFactory(uint32_t T, uint32_t t) {

    }

    void AddSmoothnessTerms(self, deriv_order=2) 
    {

        if( deriv_order == 1 ) {
            auto derivative = Pullback(SquaredNormVelocity(
                self.config_space_dim, self.dt),
                self.function_network.left_of_clique_map())
            self.function_network.register_function_for_all_cliques(
                Scale(derivative, self._velocity_scalar))
        }

        elif deriv_order == 2:
            derivative = SquaredNormAcceleration(
                self.config_space_dim, self.dt)
            self.function_network.register_function_for_all_cliques(
                Scale(derivative, self._acceleration_scalar))
        else:
            raise ValueError("deriv_order ({}) not suported".format(
                deriv_order))
        }
    }

    def add_isometric_potential_to_all_cliques(self, potential, scalar):
        """
        Apply the following euqation to all cliques:

                c(x_t) | d/dt x_t |

            The resulting Riemanian metric is isometric. TODO see paper.
            Introduced in CHOMP, Ratliff et al. 2009.
        """
        cost = Pullback(
            potential,
            self.function_network.center_of_clique_map())
        squared_norm_vel = Pullback(
            SquaredNormVelocity(self.config_space_dim, self.dt),
            self.function_network.right_of_clique_map())

        self.function_network.register_function_for_all_cliques(
            Scale(ProductFunction(cost, squared_norm_vel), scalar))

};

}