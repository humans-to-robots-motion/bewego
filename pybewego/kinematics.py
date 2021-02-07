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

from pybewego import quaternion_to_matrix
from pybewego import euler_to_quaternion
from pybewego import Robot

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
        out += " - axis : " + str(self.joint_axis_in_local) + '\n'
        out += " - local_in_prev : " + '\n' + str(self.local_in_prev) + '\n'
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

    def create_robot(self, joints):
        robot = Robot()
        for j in joints:
            body = self.joints[j]
            robot.add_rigid_body(
                body.name,
                body.joint_name,
                body.type,
                body.joint_bounds.low,
                body.joint_bounds.high,
                body.local_in_prev,
                body.joint_axis_in_local)
        return robot


class RobotConfig:

    def __init__(self, json_config):
        """
        Loads a configuration from a json file
        filename assets_data_dir() + "/baxter_right_arm.json"
        """
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
