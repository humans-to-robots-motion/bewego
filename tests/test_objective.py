#!/usr/bin/env python

# Copyright (c) 2021, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                             Jim Mainprice on Tue Feb 16 2021

import os
import sys
import numpy as np
import conftest
from pybewego import MotionObjective
from pyrieef.geometry.differentiable_geometry import *
from pyrieef.motion.trajectory import Trajectory
from pyrieef.motion.trajectory import linear_interpolation_trajectory
from pyrieef.motion.objective import smoothness_metric
from pyrieef.motion.objective import MotionOptimization2DCostMap
from numpy.testing import assert_allclose
import time


def test_motion_optimimization_hessian():

    np.random.seed(0)

    print("Check Motion Optimization (Derivatives)")

    T = 30
    n = 2
    dt = 0.01

    q_init = np.zeros(n)
    q_goal = np.ones(n)

    problem = MotionObjective(T, dt, n)
    problem.add_smoothness_terms(2, 1)
    objective = problem.objective(q_init)

    trajectory = Trajectory(T, n)
    sum_acceleration = objective.forward(trajectory.active_segment())
    print(("sum_acceleration 1 : ", sum_acceleration))

    trajectory = linear_interpolation_trajectory(q_init, q_goal, T)
    sum_acceleration = objective.forward(trajectory.active_segment())
    print(("sum_acceleration 2 : ", sum_acceleration))

    print("Test J for trajectory")
    assert check_jacobian_against_finite_difference(
        objective, True)

    # Check the hessian of the trajectory
    print("Test H for trajectory")
    is_close = check_hessian_against_finite_difference(
        objective, True, tolerance=1e-2)

    xi = np.random.rand(objective.input_dimension())
    H = objective.hessian(xi)
    H_diff = finite_difference_hessian(objective, xi)
    H_delta = H - H_diff
    print((" - H_delta dist = ", np.linalg.norm(H_delta, ord='fro')))
    print((" - H_delta maxi = ", np.max(np.absolute(H_delta))))

    assert is_close

    # Check that the hessian has the known form
    active_size = n * (trajectory.T() - 1)
    H1 = H[:active_size, :active_size]
    H2 = smoothness_metric(dt, T, n)
    H2 = H2[:active_size, :active_size]
    assert_allclose(H1, H2)


def test_motion_optimimization_compare():

    np.random.seed(0)

    print("Check Motion Optimization (Derivatives)")

    T = 30
    n = 2
    dt = 0.1

    q_init = np.zeros(n)
    q_goal = np.ones(n)

    problem1 = MotionObjective(T, dt, n)
    problem1.add_smoothness_terms(2, 20)
    objective1 = problem1.objective(q_init)

    problem2 = MotionOptimization2DCostMap(T, n)
    problem2.dt = dt
    problem2.q_init = q_init
    problem2.create_clique_network()
    problem2.set_scalars(acceleration_scalar=20 * dt)
    problem2.add_smoothness_terms(2)
    problem2.create_objective()
    objective2 = problem2.objective

    assert objective1.input_dimension() == objective2.input_dimension()

    trajectory = Trajectory(T, n)
    trajectory.set(np.random.rand(n * (T + 2)))

    cost1 = objective1.forward(trajectory.active_segment())
    cost2 = objective2.forward(trajectory.active_segment())
    print(("sum_acceleration 1 : ", cost1))
    print(("sum_acceleration 1 : ", cost2))
    assert abs(cost1 - cost2) < 1e-6

    trajectory = linear_interpolation_trajectory(q_init, q_goal, T)

    cost1 = objective1.forward(trajectory.active_segment())
    cost2 = objective2.forward(trajectory.active_segment())
    print(("sum_acceleration 2 : ", cost1))
    print(("sum_acceleration 2 : ", cost2))
    assert abs(cost1 - cost2) < 1e-6



# test_motion_optimimization_hessian()
test_motion_optimimization_compare()
