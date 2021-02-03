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
import numpy as np


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
