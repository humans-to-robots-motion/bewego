#!/usr/bin/env python

# Copyright (c) 2021
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
import numpy as np
from pybewego.kinematics import *
from pybewego import KinematicChain
from pybewego import Robot
import os


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

    def __init__(self, urdf_file, json_config=None, with_gui=False):
        if with_gui:
            # self._p = bc.BulletClient(connection_mode=pybullet.GUI,
            #                           options="--width=600 --height=400")
            pybullet.connect(
                pybullet.GUI,
                options="--width=600 --height=400")
            self._p = pybullet
        else:
            self._p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self._robot_id = self._p.loadURDF(urdf_file)
        self._njoints = self._p.getNumJoints(self._robot_id)
        self.rigid_bodies = []
        self.config = None
        if json_config is not None:
            self.config = RobotConfig(json_config)
        self._parse_rigid_bodies()

    def _euler_pyb(self, q):
        print(self._p.getEulerFromQuaternion(q))

    def _transform_pyb(self, p, q):
        R = self._p.getMatrixFromQuaternion(q)
        R = np.reshape(np.array(R), (3, 3))
        t = np.eye(4)
        t[:3, :3] = R
        t[:3, 3] = np.array(p)
        return t

    def _parse_rigid_bodies(self):
        """
        Get the joint info into a rigid body datastructure
        """
        for i in range(self._njoints):
            info = self._p.getJointInfo(self._robot_id, i)
            rigid_body = RigidBody()
            rigid_body.type = info[2]
            rigid_body.name = str(info[12], 'utf-8')
            rigid_body.joint_bounds = ScalarBounds(info[8], info[9])
            rigid_body.joint_name = str(info[1], 'utf-8')
            rigid_body.joint_axis_in_local = np.asarray(info[13])
            # print(rigid_body.name)
            append = self.config is None
            if append or (rigid_body.name in self.config.active_joint_names):
                # print_joint_info(info)
                if i > 0:
                    state = self._p.getLinkState(self._robot_id, i - 1)
                    t_com_l = self._transform_pyb(state[2], state[3])
                else:
                    t_com_l = np.eye(4)

                rigid_body.local_in_prev = t_com_l @ self._transform_pyb(
                    np.asarray(info[14]),
                    np.asarray(info[15]))
                self.rigid_bodies.append(rigid_body)

    def create_kinematics(self):
        """ Creates a Bewego kinematics object """
        robot = KinematicChain()
        for body in self.rigid_bodies:
            robot.add_rigid_body(
                body.name,
                body.joint_name,
                body.type,
                body.joint_bounds.low,
                body.joint_bounds.high,
                body.local_in_prev,
                body.joint_axis_in_local)
        return robot

    def create_robot(self, keypoints):
        """ Creates a Bewego robot object (which includes task maps) """
        kinematic_structure_info = []
        for body in self.rigid_bodies:
            body_info = RigidBodyInfo()
            body_info.name = body.name
            body_info.joint_name = body.joint_name
            body_info.joint_type = body.type
            body_info.dof_lower_limit = body.joint_bounds.low
            body_info.dof_upper_limit = body.joint_bounds.high
            body_info.local_in_prev = body.local_in_prev
            body_info.joint_axis_in_local = body.joint_axis_in_local
            kinematic_structure_info.append(body_info)
        return Robot(kinematic_structure_info, keypoints)

    def sample_config(self):
        """ Samples a configuration in dof bounds for the active DoFs """
        q = np.empty(len(self.rigid_bodies))
        for i, body in enumerate(self.rigid_bodies):
            l = body.joint_bounds.low
            h = body.joint_bounds.high
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

    def set_and_update(self, q, joint_ids=None):
        if joint_ids is None:
            joint_ids = range(self._njoints)
        q = np.asarray(q).reshape(len(joint_ids), 1)
        self._p.resetJointStatesMultiDof(self._robot_id, joint_ids, q)

    def configuration(self):
        return np.asarray([i[0] for i in self._p.getJointStates(
            self._robot_id, range(self._njoints))])

    def transform(self, idx):
        """
        Returns the tranformation matrix of link origin in world coordinates

        @param idx joint id

        t_com_w : position of COM in world cooridnates
        t_com_l : position of COM in world cooridnates

        # t_com_l = transform(state[2], state[3])
        # print("t_com_l (1) : \n", t_com_l)
        # return transform(state[0], state[1]) @ np.linalg.inv(t_com_l)

        @return Rotation matrix
        """
        state = self._p.getLinkState(self._robot_id, idx)
        return transform(state[4], state[5])

    def rotation(self, idx):
        """
        Returns the rotation matrix of link origin in world coordinates

        @param idx joint id

        q_com_w : rotation of COM in world cooridnates
        q_com_l : rotation of COM in world cooridnates

        @return Rotation matrix
        """
        # info = self._p.getJointInfo(self._robot_id, idx)
        # q_joint = info[15]
        # R_joint = self._p.getMatrixFromQuaternion(q_joint)
        # R_joint = np.reshape(np.array(R_joint), (3, 3))
        # print(R_joint)

        state = self._p.getLinkState(self._robot_id, idx)
        q_com_w = state[1]
        R_com_w = self._p.getMatrixFromQuaternion(q_com_w)
        R_com_w = np.reshape(np.array(R_com_w), (3, 3))
        q_com_l = state[3]
        R_com_l = self._p.getMatrixFromQuaternion(q_com_l)
        R_com_l = np.reshape(np.array(R_com_l), (3, 3))
        return R_com_w @ R_com_l.T

    def position(self, idx):
        """
        Returns the position of link origin in world coordinates

        @param idx joint id
        @return 3d numpy array
        """
        return self.transform(idx)[:3, 3].T

    def jacobian_pos(self, idx):
        com = [0., 0., 0.]
        q, dq, tau = self.get_motor_joint_states()
        jac = self._p.calculateJacobian(self._robot_id, idx, com, q, dq, tau)
        return np.array(jac)[0]
