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

from pybewego import quaternion_to_matrix
from pybewego import euler_to_quaternion
from pybewego import KinematicChain

from pyrieef.geometry.differentiable_geometry import *

import numpy as np
import xml.etree.ElementTree as ET
import json
import os


def assets_data_dir():
    return os.path.abspath(os.path.dirname(__file__)) + os.sep + "../data"


def transform(pos, quat):
    """
     Creates a numpy matrix from a position and quaternion
     specifying a transform
    """
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos)
    T[:3, :3] = quaternion_to_matrix(quat)
    return T


class ScalarBounds:

    def __init__(self, low, high):
        self.low = low
        self.high = high


class RigidBody:

    def __init__(self):

        self.name = ""                          # Rigid body name in URDF.
        self.joint_name = ""                    # Joint body name
        self.type = ""                          # Joint type
        self.parent = ""                        # Parent body name
        self.joint_bounds = ScalarBounds(0, 0)  # Parent joint limits
        self.frame_in_base = np.eye(4)          # FK solution
        self.frame_in_local = np.eye(4)         # Joint transformed frame
        self.local_in_prev = np.eye(4)          # Prev to curr transform

        # Info about the joint axis.
        # If it's axis aligned, then we can skip some computation.
        self.joint_axis_in_local = np.array([0, 0, 0])
        self.joint_axis_in_base = np.array([0, 0, 0])

    def set_type(self, t):
        if t == "revolute":
            self.type = 0
        elif t == "prismatic":
            self.type = 1
        elif t == "fixed":
            self.type = 4

    def __str__(self):
        out = ""
        out += " - joint name : " + str(self.joint_name) + '\n'
        out += " - body name : " + str(self.name) + '\n'
        out += " - parent : " + str(self.parent) + '\n'
        out += " - type : " + str(self.type) + '\n'
        out += " - limit (l) : " + str(self.joint_bounds.low) + '\n'
        out += " - limit (h) : " + str(self.joint_bounds.high) + '\n'
        out += " - local_in_prev : " + '\n' + str(self.local_in_prev) + '\n'
        out += " - axis : " + str(self.joint_axis_in_local) + '\n'
        return out


class Kinematics:

    def __init__(self, urdf_file):
        urdf_root = ET.parse(urdf_file).getroot()
        self.links = {}
        self.joints = {}
        jointid = 0
        for child in urdf_root:
            if child.tag == "link":
                self.links[child.attrib["name"]] = {
                    "childs": [], "parents": []}
            elif child.tag == "joint":
                joint = RigidBody()
                joint.set_type(child.attrib["type"])
                joint.joint_name = child.attrib["name"]
                joint.id = jointid
                for jel in child:
                    if jel.tag == "parent":
                        joint.parent = jel.attrib["link"]
                        self.links[jel.attrib["link"]][
                            "childs"].append(child.attrib["name"])
                    elif jel.tag == "child":
                        joint.name = jel.attrib["link"]
                        if joint.name in self.links:
                            self.links[jel.attrib["link"]][
                                "parents"].append(child.attrib["name"])
                    elif jel.tag == "origin":
                        splits = jel.attrib["xyz"].split()
                        position = np.array([
                            float(splits[0]),
                            float(splits[1]),
                            float(splits[2])])
                        quaternion = np.array([0, 0, 0, 1])
                        if "rpy" in jel.attrib:
                            splits = jel.attrib["rpy"].split()
                            quaternion = euler_to_quaternion(np.array([
                                float(splits[0]),
                                float(splits[1]),
                                float(splits[2])]))
                        joint.local_in_prev = transform(position, quaternion)
                    elif jel.tag == "limit":
                        joint.joint_bounds.low = float(jel.attrib["lower"])
                        joint.joint_bounds.high = float(jel.attrib["upper"])
                    elif jel.tag == "axis":
                        splits = jel.attrib["xyz"].split()
                        joint.joint_axis_in_local = np.array([
                            float(splits[0]),
                            float(splits[1]),
                            float(splits[2])])
                    else:
                        # print('WARNING: Tag "' + jel.tag + '" not supported')
                        pass
                self.joints[joint.name] = joint
                # print(joint)
        for l in self.links:
            if self.links[l]["parents"] == []:
                self.baselink = l

    def print_kinematics_info(self):
        for idx, name in enumerate(self.joints):
            print("body ({}) ->\n{}".format(idx, self.joints[name]))

    def create_robot(self, active_bodies):
        kinematic_chain = KinematicChain()
        for name in active_bodies:
            body = self.joints[name]
            kinematic_chain.add_rigid_body(
                body.name,
                body.joint_name,
                body.type,
                body.joint_bounds.low,
                body.joint_bounds.high,
                body.local_in_prev,
                body.joint_axis_in_local)
        return kinematic_chain


class RobotConfig:
    """
    Loads a configuration from a json file
    filename assets_data_dir() + "/baxter_right_arm.json"
    """

    def __init__(self, json_config):

        filename = assets_data_dir() + os.sep + json_config
        with open(filename, "r") as read_file:
            config = json.loads(read_file.read())
            self.name = config["name"]
            self.keypoints = config["keypoints"]
            self.active_joint_names = config["joint_names"]
            self.active_joint_ids = config["joint_ids"]
            self.active_dofs = config["active_dofs"]
            self.scale = config["scale"]
            self.base_joint_id = config["base_joint_id"]
            self.end_effector_id = config["end_effector_id"]


class ForwardKinematics(DifferentiableMap):
    """ 
    Simple forward kinematics as a differentiable map

        This class allows to test the kinematic Jacobians
        against finite differences.

    Parameters
    ----------
        robot : kinematics
        eef_id : End Effector ID
                 Example, dof od for pybewego or joint id for pybullet
        dofs : list of dof ids
        output : position, axis, frame
        subset : subset of ids that to take in the Jacobian
        axis : x,y,z
    """

    def __init__(self, robot, eef_id, dofs,
                 output="position",
                 subset=None,
                 axis="x"):
        self._robot = robot
        self._eef_id = eef_id
        self._dofs = dofs
        self._n = len(dofs)
        self._subset = subset
        self._output = output

        if self._output == "position":
            self._m = 3
        elif self._output == "axis":
            self._m = 3
            axes = {
                "x": 0,
                "y": 1,
                "z": 2
            }
            self._axis = axes[axis]
        elif self._output == "frame":
            self._m = 12

    def output_dimension(self):
        return self._m

    def input_dimension(self):
        return self._n

    def set_and_update(self, q):
        if self._subset is not None:
            self._robot.set_and_update(q, self._dofs)
        else:
            self._robot.set_and_update(q)

    def get_jacobian(self):
        if self._output == "position":
            J = self._robot.jacobian_pos(self._eef_id)
        elif self._output == "axis":
            J = self._robot.jacobian_axis(self._eef_id, self._axis)
        elif self._output == "frame":
            J = self._robot.jacobian_frame(self._eef_id)

        if self._subset is None:
            return J
        return J[:, self._subset]

    def forward(self, q):
        self.set_and_update(q)
        if self._output == "position":
            return self._robot.position(self._eef_id)
        elif self._output == "axis":
            t = self._robot.transform(self._eef_id)
            return t[:3, self._axis]
        elif self._output == "frame":
            t = self._robot.transform(self._eef_id)
            return np.hstack([t[:3, 3], t[:3, 0], t[:3, 1], t[:3, 2]])

    def jacobian(self, q):
        self.set_and_update(q)
        return self.get_jacobian()
