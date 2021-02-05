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
import json
import os


def assets_data_dir():
    return os.path.abspath(os.path.dirname(__file__)) + os.sep + "../data"


def print_joint_info(info):
    print("Joint info: " + str(info))
    print(" - joint name : ", info[1])      # name of the joint in URDF
    print(" - type : ", info[2])            # p.JOINT_REVOLUTE or p.PRISMATIC
    print(" - q idx : ", info[3])           # Index in config
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

    def __init__(self, urdf_file, json_config=None):
        self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self._robot_id = self._p.loadURDF(urdf_file)
        self._njoints = self._p.getNumJoints(self._robot_id)
        print("number joints: " + str(self._njoints))
        self.rigid_bodies = []
        self.active_joint_names = None
        if json_config is not None:
            self._load_config_from_file(json_config)
        self._parse_rigid_bodies()

    def _load_config_from_file(self, json_config):
        """
        Loads a configuration from a json file
        filename assets_data_dir() + "/baxter_right_arm.json"
        """
        filename = assets_data_dir() + os.sep + json_config
        with open(filename, "r") as read_file:
            config = json.loads(read_file.read())
            self.config_name = config["name"]
            self.keypoints = config["keypoints"]
            self.active_joint_names = config["joint_names"]
            self.active_joint_ids = config["joint_ids"]
            self.active_dofs = config["active_dofs"]
            self.scale = config["scale"]
            self.base_joint_id = config["base_joint_id"]
            self.end_effector_id = config["end_effector_id"]
        print(self.active_dofs)

    def _parse_rigid_bodies(self):
        """
        Get the joint info into a rigid body datastructure
        """
        for i in range(self._njoints):
            info = self._p.getJointInfo(self._robot_id, i)
            print("joint id : ", i)
            print_joint_info(info)
            rigid_body = RigidBody()
            rigid_body.type = info[2]
            rigid_body.name = str(info[12], 'utf-8')
            rigid_body.joint_bounds = ScalarBounds(info[8], info[9])
            rigid_body.joint_name = str(info[1], 'utf-8')
            rigid_body.joint_axis_in_local = np.asarray(info[13])
            rigid_body.local_in_prev = transform(
                np.asarray(info[14]),
                np.asarray(info[15])
            )
            # print(rigid_body.name)
            append = self.active_joint_names is None
            if append or (rigid_body.name in self.active_joint_names):
                self.rigid_bodies.append(rigid_body)

    def create_robot(self):
        robot = Robot()
        for body in self.rigid_bodies:
            robot.add_rigid_body(
                body.name,
                body.joint_name,
                body.type,
                body.joint_bounds.lower,
                body.joint_bounds.upper,
                body.local_in_prev,
                body.joint_axis_in_local)
        return robot

    def sample_config(self):
        """ Samples a configuration in dof bounds for the active DoFs """
        q = np.array(len(self.rigid_bodies))
        for i, body in enumerate(self.rigid_bodies):
            l = body.joint_bounds.lower
            h = body.joint_bounds.upper
            q[i] = np.random.uniform(low=l, high=h)
        return q

    def get_motor_joint_states(self):
        joint_states = self._p.getJointStates(
            self._robot_id, range(self._njoints))
        joint_infos = [self._p.getJointInfo(self._robot_id, i)
                       for i in range(self._njoints)]
        joint_states = [j for j, i in zip(
            joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def set_and_update(self, q, dofs=range(self._njoints)):
        q = np.asarray(q).reshape(self._njoints, 1)
        self._p.resetJointStatesMultiDof(self._robot_id, dofs, q)

    def get_configuration(self):
        return np.asarray([i[0] for i in self._p.getJointStates(
            self._robot_id, range(self._njoints))])

    def get_transform(self, idx):
        """
        Returns the tranformation matrix of link origin in world coordinates

        @param idx joint id

        t_com_w : position of COM in world cooridnates
        t_com_l : position of COM in world cooridnates

        @return Rotation matrix
        """
        state = self._p.getLinkState(self._robot_id, idx)
        t_com_l = transform(state[2], state[3])
        return transform(state[0], state[1]) @ np.linalg.inv(t_com_l)

    def get_rotation(self, idx):
        """
        Returns the rotation matrix of link origin in world coordinates

        @param idx joint id

        q_com_w : rotation of COM in world cooridnates
        q_com_l : rotation of COM in world cooridnates

        @return Rotation matrix
        """
        state = self._p.getLinkState(self._robot_id, idx)
        q_com_w = state[1]
        R_com_w = self._p.getMatrixFromQuaternion(q)
        R_com_w = np.reshape(np.array(R), (3, 3))
        q_com_l = state[3]
        R_com_l = self._p.getMatrixFromQuaternion(q)
        R_com_l = np.reshape(np.array(R), (3, 3))
        return R_com_w @ R_com_l.T

    def get_position(self, idx):
        """
        Returns the position of link origin in world coordinates

        @param idx joint id
        @return 3d numpy array
        """
        return self.get_transform(idx)[:3, 3].T

    def get_jacobian(self, idx):
        com = [0., 0., 0.]
        q, dq, tau = self.get_motor_joint_states()
        jac = np.array(self._p.calculateJacobian(
            self._robot_id, idx, com, q, dq, tau)[0])
        return jac
