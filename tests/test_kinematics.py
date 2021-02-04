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
#                                            Jim Mainprice on Sat Jun 6 2020

# import numpy as np
from test_imports import *
from pybewego.pybullet_loader import *
from numpy.testing import assert_allclose
import time


DATADIR = "../../pybullet_robots/data/"


def test_import_planar():
    PybulletRobot("../data/r2_robot.urdf")


def test_pybullet_forward_kinematics():
    robot = PybulletRobot("../data/r2_robot.urdf")
    q = [np.pi / 2, np.pi / 2, np.pi]
    robot.set_and_update(q)
    robot.set_and_update(np.array(q))
    q_bullet = robot.get_configuration()
    print(q_bullet)
    for i, q_i in enumerate(q):
        assert q_i == q_bullet[i]
    p = robot.get_position(2)
    assert_allclose(p[:2], [-1, 1], atol=1e-6)
    print("p : ", p)


def test_bewego_forward_kinematics():
    robot = PybulletRobot("../data/r2_robot.urdf").create_robot()
    q = [np.pi / 2, np.pi / 2, 0]
    robot.set_and_update(q)
    p = robot.get_position(2)
    assert_allclose(p[:2], [-1, 1], atol=1e-6)
    print("p : ", p)


def test_random_forward_kinematics():
    np.random.seed(0)
    configurations = np.random.uniform(low=-3.14, high=3.14, size=(100, 3))
    r1 = PybulletRobot("../data/r2_robot.urdf")
    r2 = r1.create_robot()
    for q in configurations:
        r1.set_and_update(q)
        p1 = r1.get_position(2)
        r2.set_and_update(q)
        p2 = r2.get_position(2)
        assert_allclose(p1[:2], p2[:2], atol=1e-6)

    # bewego rootine is 5 ~ 10 X faster than pybullet
    # for this the code has to be compiled in Release or RelWithDebInfo.

    t0 = time.time()
    for q in configurations:
        r1.set_and_update(q)
        p1 = r1.get_position(2)
    print("time 1 : ", time.time() - t0)

    t0 = time.time()
    for q in configurations:
        r2.set_and_update(q)
        p2 = r2.get_position(2)
    print("time 2 : ", time.time() - t0)


def test_jacobian():
    r1 = PybulletRobot("../data/r2_robot.urdf")
    r2 = r1.create_robot()

    q = [1, 2, 0]

    r1.set_and_update(q)
    J = r1.get_jacobian(2)
    print("Jacobian (r1) : \n", J)

    r2.set_and_update(q)
    J = r2.get_jacobian(2)
    print("Jacobian (r2) : \n", J)

    np.random.seed(0)
    configurations = np.random.uniform(low=-3.14, high=3.14, size=(100, 3))
    for q in configurations:
        r1.set_and_update(q)
        p1 = r1.get_jacobian(2)
        r2.set_and_update(q)
        p2 = r2.get_jacobian(2)
        assert_allclose(p1[:2], p2[:2], atol=1e-6)

    t0 = time.time()
    for q in configurations:
        r1.set_and_update(q)
        p1 = r1.get_jacobian(2)
    print("time 1 : ", time.time() - t0)

    t0 = time.time()
    for q in configurations:
        r2.set_and_update(q)
        p2 = r2.get_jacobian(2)
    print("time 2 : ", time.time() - t0)


def test_jacobian_baxter():
    robot = PybulletRobot(
        DATADIR + "baxter_common/baxter_description/urdf/toms_baxter.urdf")
    # robot.set_and_update()
    print(len(robot.get_configuration()))
    robot.set_and_update([0] * 56)
    J = robot.get_jacobian(20)
    print(J.shape)


# test_pybullet_forward_kinematics()
# test_bewego_forward_kinematics()
# test_random_forward_kinematics()
# test_jacobian()
test_jacobian_baxter()
