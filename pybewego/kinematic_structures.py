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
import numpy as np
import xml.etree.ElementTree as ET


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
        self.type = ""                          # Joint type
        self.joint_bounds = ScalarBounds(0, 0)  # Parent joint limits
        self.joint_name = ""                    # Joint body name
        self.frame_in_base = np.eye(4)          # FK solution
        self.frame_in_local = np.eye(4)         # Joint transformed frame
        self.local_in_prev = np.eye(4)          # Prev to curr transform

        # Info about the joint axis.
        # If it's axis aligned, then we can skip some computation.
        self.joint_axis_in_local = np.array([0, 0, 0])
        self.joint_axis_in_base = np.array([0, 0, 0])

    def __str__(self):
        out = ""
        out += " - joint name : " + str(self.joint_name) + '\n'
        out += " - type : " + str(self.type) + '\n'
        out += " - limit (l) : " + str(self.joint_bounds.low) + '\n'
        out += " - limit (h) : " + str(self.joint_bounds.high) + '\n'
        out += " - body name : " + str(self.name) + '\n'
        out += " - axis : " + str(self.joint_axis_in_local) + '\n'
        out += " - local_in_prev : " + '\n' + str(self.local_in_prev) + '\n'
        return out


# class Joint:

#     def __init__(self):
#         self.parent = ""
#         self.child = ""
#         self.origin = tf.constant([0., 0., 0.])
#         self.axis = (0, 0, 1)
#         self.id = 0
#         self.type = "revolute"

#     def __str__(self):
#         out = "parent: " + self.parent + ", child: " + self.child
#         out += ", origin: " + str(self.origin) + ", axis: " + str(self.axis)
#         return out


class Kinematics:

    def __init__(self,
                 urdf_file=None,
                 data_fixed=None):
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
                joint.type = child.attrib["type"]
                joint.name = child.attrib["name"]
                joint.id = jointid
                for jel in child:
                    if jel.tag == "parent":
                        joint.parent = jel.attrib["link"]
                        self.links[jel.attrib["link"]][
                            "childs"].append(child.attrib["name"])
                    elif jel.tag == "child":
                        joint.child = jel.attrib["link"]
                        self.links[jel.attrib["link"]][
                            "parents"].append(child.attrib["name"])
                    elif jel.tag == "origin":
                        splits = jel.attrib["xyz"].split()
                        origin_p = np.array([
                            float(splits[0]),
                            float(splits[1]),
                            float(splits[2])])
                        splits = jel.attrib["rpy"].split()
                        origin_r = np.array([
                            float(splits[0]),
                            float(splits[1]),
                            float(splits[2])])
                    elif jel.tag == "axis":
                        splits = jel.attrib["xyz"].split()
                        joint.axis = np.array([
                            float(splits[0]),
                            float(splits[1]),
                            float(splits[2])])
                    else:
                        #print('WARNING: Tag "' + jel.tag + '" not supported')
                        pass
                self.joints[child.attrib["name"]] = joint
        for l in self.links:
            if self.links[l]["parents"] == []:
                self.baselink = l

    #@tf.function
    def forward(self, jointstates):
        pos = tf.constant([0., 0., 0.])
        rot = tf.constant([0., 0., 0., 1.])
        link = self.baselink
        link_states = []

        def computeChilds(link, pos, rot):
            for jointname in self.links[link]["childs"]:
                joint = self.joints[jointname]
                newlink = joint.child
                if joint.id >= 0:
                    if joint.type == "revolute":
                        newpos = pos + tfg.quaternion.rotate(joint.origin, rot)
                        newrot = tfg.quaternion.multiply(
                            rot, tfg.quaternion.from_euler(joint.axis * jointstates[joint.id]))
                    elif joint.type == "prismatic":
                        newpos = pos + tfg.quaternion.rotate(joint.origin, rot) + tfg.quaternion.rotate(
                            joint.axis * jointstates[joint.id], rot)
                        newrot = rot
                else:
                    if joint.type == "revolute":
                        newpos = pos + tfg.quaternion.rotate(joint.origin, rot)
                        newrot = tfg.quaternion.multiply(
                            rot, tfg.quaternion.from_euler(joint.axis * joint.jointstate))
                    elif joint.type == "prismatic":
                        newpos = pos + tfg.quaternion.rotate(
                            joint.origin, rot) + tfg.quaternion.rotate(joint.axis * joint.jointstate, rot)
                        newrot = rot
                link_states.append((newlink, newpos, newrot))
                computeChilds(newlink, newpos, newrot)

        computeChilds(link, pos, rot)

        return link_states
