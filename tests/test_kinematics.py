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

import os
import sys
import numpy as np
import conftest
from pybewego import *
from pybewego.pybullet_loader import *
from pybewego.kinematics import *
from pyrieef.geometry.differentiable_geometry import *
from numpy.testing import assert_allclose
import time


directory = os.path.abspath(os.path.dirname(__file__))
DATADIR = directory + "/../../pybullet_robots/data/"


def test_import_planar():
    PybulletRobot("../data/r2_robot.urdf")


def test_import_planar():
    kinematics = Kinematics("../data/r3_robot.urdf")
    kinematics.print_kinematics_info()


def test_geometry():
    robot = PybulletRobot("../data/r2_robot.urdf")
    rpy = np.random.random(3)
    q1 = robot._p.getQuaternionFromEuler(rpy)
    q2 = euler_to_quaternion(rpy)
    assert_allclose(q1, q2, atol=1e-6)


def test_parser():
    urdf = "../data/r2_robot.urdf"
    urdf = DATADIR + "baxter_common/baxter_description/urdf/toms_baxter.urdf"
    kinematics = Kinematics(urdf)


def test_pybullet_forward_kinematics():
    robot = PybulletRobot("../data/r2_robot.urdf")
    q = [np.pi / 2, np.pi / 2, np.pi]
    robot.set_and_update(q)
    robot.set_and_update(np.array(q))
    q_bullet = robot.configuration()
    print(q_bullet)
    for i, q_i in enumerate(q):
        assert q_i == q_bullet[i]
    p = robot.position(2)
    assert_allclose(p[:2], [-1, 1], atol=1e-6)
    print("p : ", p)


def test_bewego_forward_kinematics():
    robot = PybulletRobot("../data/r2_robot.urdf").create_robot()
    q = [np.pi / 2, np.pi / 2, 0]
    robot.set_and_update(q)
    p = robot.position(2)
    assert_allclose(p[:2], [-1, 1], atol=1e-6)
    print("p : ", p)


def test_random_forward_kinematics():
    np.random.seed(0)
    configurations = np.random.uniform(low=-3.14, high=3.14, size=(100, 4))
    r1 = PybulletRobot("../data/r3_robot.urdf")
    r2 = r1.create_robot()
    for q in configurations:
        r1.set_and_update(q)
        p1 = r1.position(3)
        r2.set_and_update(q)
        p2 = r2.position(3)
        assert_allclose(p1[:2], p2[:2], atol=1e-6)

    # bewego rootine is 5 ~ 10 X faster than pybullet
    # for this the code has to be compiled in Release or RelWithDebInfo.

    t0 = time.time()
    for q in configurations:
        r1.set_and_update(q)
        p1 = r1.position(3)
    print("time 1 : ", time.time() - t0)

    t0 = time.time()
    for q in configurations:
        r2.set_and_update(q)
        p2 = r2.position(3)
    print("time 2 : ", time.time() - t0)


def test_jacobian():
    r1 = PybulletRobot("../data/r2_robot.urdf")
    r2 = r1.create_robot()

    q = [1, 2, 0]

    r1.set_and_update(q)
    J = r1.jacobian_pos(2)
    print("Jacobian (r1) : \n", J)

    r2.set_and_update(q)
    J = r2.jacobian_pos(2)
    print("Jacobian (r2) : \n", J)

    np.random.seed(0)
    configurations = np.random.uniform(low=-3.14, high=3.14, size=(100, 3))
    for q in configurations:
        r1.set_and_update(q)
        J1 = r1.jacobian_pos(2)
        r2.set_and_update(q)
        J2 = r2.jacobian_pos(2)
        assert_allclose(J1, J2, atol=1e-6)

    t0 = time.time()
    for q in configurations:
        r1.set_and_update(q)
        p1 = r1.jacobian_pos(2)
    print("time 1 : ", time.time() - t0)

    t0 = time.time()
    for q in configurations:
        r2.set_and_update(q)
        p2 = r2.jacobian_pos(2)
    print("time 2 : ", time.time() - t0)


def test_forward_kinematics_baxter():
    urdf = DATADIR + "baxter_common/baxter_description/urdf/toms_baxter.urdf"
    r1 = PybulletRobot(urdf, "baxter_right_arm.json")
    r1.set_and_update([0] * r1._njoints)
    base = r1.transform(r1.config.base_joint_id)

    kinematics = Kinematics(urdf)
    r2 = kinematics.create_robot(r1.config.active_joint_names)
    r2.set_base_transform(base)

    configurations = [None] * 100
    for i in range(len(configurations)):
        configurations[i] = r1.sample_config()

    for q in configurations:
        np.set_printoptions(suppress=True)
        r1.set_and_update(q, r1.config.active_joint_ids)
        r2.set_and_update(q)
        for dof_idx, joint_idx in zip(range(6), r1.config.active_joint_ids):
            p1 = r1.transform(joint_idx)
            p2 = r2.transform(dof_idx)
            assert_allclose(p1, p2, atol=1e-6)

    t0 = time.time()
    for q in configurations:
        r1.set_and_update(q, r1.config.active_joint_ids)
        p1 = r1.transform(6)
    print("time 1 : ", time.time() - t0)

    t0 = time.time()
    for q in configurations:
        r2.set_and_update(q)
        p2 = r2.transform(6)
    print("time 2 : ", time.time() - t0)


def test_differentiable_jacobian():
    urdf = DATADIR + "baxter_common/baxter_description/urdf/toms_baxter.urdf"
    config = RobotConfig("baxter_right_arm.json")
    kinematics = Kinematics(urdf)
    robot = kinematics.create_robot(config.active_joint_names)
    robot.set_base_transform(np.eye(4))
    output_options = ["position", "axis", "frame"]
    for o in output_options:
        for link_id in range(7):
            if o == "axis":
                for k in ["x", "y", "z"]:
                    fk = ForwardKinematics(
                        robot, link_id, range(7),
                        output=o,
                        axis=k)
                    assert check_jacobian_against_finite_difference(
                        fk, verbose=False)
            else:
                fk = ForwardKinematics(
                    robot, link_id, range(7),
                    output=o)
                assert check_jacobian_against_finite_difference(
                    fk, verbose=False)
    print("FK OK!")


def test_jacobian_baxter():
    urdf = DATADIR + "baxter_common/baxter_description/urdf/toms_baxter.urdf"

    r1 = PybulletRobot(urdf, "baxter_right_arm.json")
    print(len(r1.configuration()))
    r1.set_and_update([0] * 56)
    base = r1.transform(r1.config.base_joint_id)

    kinematics = Kinematics(urdf)
    r2 = kinematics.create_robot(r1.config.active_joint_names)
    r2.set_base_transform(base)

    np.set_printoptions(suppress=True)

    fk = ForwardKinematics(r2, 6, range(7))
    assert check_jacobian_against_finite_difference(
        fk, verbose=False)

    fk = ForwardKinematics(r1, 19,
                           dofs=r1.config.active_joint_ids,
                           subset=r1.config.active_dofs)
    assert check_jacobian_against_finite_difference(
        fk, verbose=False, tolerance=1e-3)

    r1.set_and_update([0] * 56)
    r2.set_and_update([0] * 7)

    J1 = r1.jacobian_pos(19)[:, r1.config.active_dofs]
    J2 = r2.jacobian_pos(6)

    assert_allclose(J1, J2, atol=1e-6)


test_import_planar()
# test_geometry()
# test_parser()
# test_pybullet_forward_kinematics()
# test_bewego_forward_kinematics()
# test_random_forward_kinematics()
# test_jacobian()
# test_forward_kinematics_baxter()
# test_differentiable_jacobian()
# test_jacobian_baxter()
