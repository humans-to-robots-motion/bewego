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
#include <bewego/derivatives/differentiable_map.h>
#include <bewego/numerical_optimization/ipopt_problem.h>
#include <bewego/numerical_optimization/trajectory_optimization.h>
#include <bewego/util/misc.h>
#include <gtest/gtest.h>

using namespace bewego;
using namespace bewego::numerical_optimization;
using namespace std;
using std::cout;
using std::endl;

TEST_F(TrajectoryQCQPTest, OptimizeMotionIpopt) {
  bool verbose = false;
  set_verbose(verbose);

  for (uint32_t i = 0; i < 100; i++) {
    IpoptOptimizer constrained_optimizer;
    constrained_optimizer.set_verbose(verbose);
    // constrained_optimizer.set_option("derivative_test", "first-order");
    // constrained_optimizer.set_option("derivative_test", "second-order");
    // constrained_optimizer.set_option("derivative_test_tol", 1e-4);
    constrained_optimizer.set_option("constr_viol_tol", 1e-7);

    ConstrainedSolution solution = constrained_optimizer.Run(
        *nonlinear_problem_, initial_solution_.ActiveSegment());
    ASSERT_TRUE(solution.success());
    ValidateSolution(solution);
  }
}
