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
#                                    Jim Mainprice on Wednesday February 3 2021


import pybullet_utils.bullet_client as bc
import pybullet
from kinematic_structures import ScalarBounds
from kinematic_structures import RigidBody
from kinematic_structures import transform
import numpy as np
from pybewego import Robot


def print_joint_info(info):
    print("Joint info: " + str(info))
    print(" - joint name : ", info[1])      # name of the joint in URDF
    print(" - type : ", info[2])            # p.JOINT_REVOLUTE or p.PRISMATIC
    print(" - limit (l) : ", info[8])       # lower limit
    print(" - limit (h) : ", info[9])       # higher limit
    print(" - body name : ", info[12])
    print(" - axis : ", info[13])           # joint axis
    print(" - origin (p) : ", info[14])     # position 3d
    print(" - origin (r) : ", info[15])     # quaternion 4d


class PybulletRobot:

    """
    Robot that is based off of a pybullet object
    Parses the URDF using pybullet
    """

    def __init__(self, filename):
        self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self._robot_id = self._p.loadURDF(filename)
        self.rigid_bodies = []

        # load robot
        self._njoints = self._p.getNumJoints(self._robot_id)
        print("number joints: " + str(self._njoints))
        for i in range(self._njoints):
            info = self._p.getJointInfo(self._robot_id, i)
            # print_joint_info(info)
            rigid_body = RigidBody()
            rigid_body.name = info[12]
            rigid_body.joint_bounds = ScalarBounds(info[8], info[9])
            rigid_body.joint_name = info[1]
            rigid_body.joint_axis_in_local = np.asarray(info[13])
            rigid_body.local_in_prev = transform(
                np.asarray(info[14]),
                np.asarray(info[15])
            )
            print(rigid_body)
            self.rigid_bodies.append(rigid_body)

    def create_robot(self):
        robot = Robot()
        for body in self.rigid_bodies:
            robot.add_rigid_body(
                body.name,
                body.joint_name,
                body.local_in_prev,
                body.joint_axis_in_local)
        return robot

    def set_and_update(self, q):
        assert len(q) == self._njoints
        q = np.asarray(q).reshape(self._njoints, 1)
        self._p.resetJointStatesMultiDof(
            self._robot_id, range(self._njoints), q)

    def get_configuration(self):
        return np.asarray([i[0] for i in self._p.getJointStates(
            self._robot_id, range(self._p.getNumJoints(self._robot_id)))])

    def get_position(self, idx):
        return np.array(self._p.getLinkState(self._robot_id, idx)[0])

    def get_rotation(self, idx):
        link_rot = self._p.getLinkState(self._robot_id, idx)[1]
        R = self._p.getMatrixFromQuaternion(link_rot)
        return np.reshape(np.array(R, (3, 3)))

    def get_jacobian(self, idx):
        q = list(self.get_configuration())
        zero_vec = [0.] * len(q)
        jac = np.array(self._p.calculateJacobian(self._robot_id, idx, [
                       0., 0., 0.], q, zero_vec, zero_vec)[0])
        return jac
