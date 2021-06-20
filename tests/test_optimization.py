#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
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
#                                            Jim Mainprice on Mon Mar 1 2021

import os
import sys
import conftest
import cvxopt
import numpy as np
from numpy.testing import assert_allclose
from pybewego import test_motion_optimization
from pybewego import RobotOptimizer
from pybewego.kinematics import *


def load_qp():

    directory = os.path.abspath(os.path.dirname(__file__)) + os.sep + "qp_data"

    # Objective
    H = np.loadtxt(directory + os.sep + "Hm.csv")  # P
    d = np.loadtxt(directory + os.sep + "dv.csv")  # q
    c = np.loadtxt(directory + os.sep + "c.csv")

    # Inequalities
    A = np.loadtxt(directory + os.sep + "Am.csv")  # G
    a = np.loadtxt(directory + os.sep + "av.csv")  # h
    print(A.shape)

    # Equalities
    B = np.loadtxt(directory + os.sep + "Bm.csv")  # A
    b = np.loadtxt(directory + os.sep + "bv.csv")  # b

    return H, d, A, a, B, b


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    print("z : ", sol['z'])
    print("y : ", sol['y'])
    print("s : ", np.array(sol['s']))  # Inequality
    print(sol)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def test_cvox():
    """
    The idea is to setup a QP that we can test 
    in the C++ implementation side. The numbers here come
    from cvxopt itself, and have been hard-coded in the
    C++ implementation test suit.
    """
    P, q, G, h, A, b = load_qp()
    x = cvxopt_solve_qp(P, q, G, h, A, b)
    x_expected = np.array([
        -1.10833837, 2.71244381, 1.57379586, 1.34409748, -0.98735542,
        -1.9763331, -0.32983447, -0.11374623, 0.89995774, -2.38376993])
    assert_allclose(x, x_expected)
    print("x_opt (py)  : ", x)
    print("x_opt (cpp) : ", x_expected)


def test_many_optimization():

    for _ in range(100):
        assert test_motion_optimization()


def test_optimizer_construction():
    """
        - I need to create a robot using the bodies info
        - check if I can do that for Baxter using the python inteface
        - First do it for the 3DOF planar robot.
    """
    kinematics = Kinematics(urdf_file="../data/r3_robot.urdf")
    active_dofs = ["link1", "link2", "link3"]
    keypoints = [("link3", .01)]
    robot = kinematics.create_robot(active_dofs, keypoints)
    optimizer = RobotOptimizer(3, 30, .1, [0., 1., 0., 1., 0., 1.], robot)

# test_cvox()
test_many_optimization()
