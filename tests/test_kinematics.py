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


def test_jacobian():
    robot = PybulletRobot("../data/r2_robot.urdf")
    q = [2, 1, 3]
    robot.set_and_update(q)
    robot.get_jacobian(2)
    print("jacobian ok!")


test_pybullet_forward_kinematics()
test_bewego_forward_kinematics()
# test_jacobian()
