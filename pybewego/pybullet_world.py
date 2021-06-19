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
#                                         Jim Mainprice on Friday June 18 2021

import pybullet
from pyrieef.geometry.workspace import *


class PybulletWorld:

    """
    World represents a workspace and robot
    based on pybullet rendering
    """

    def __init__(self, robot):

        self._p = robot._p

        # create basic environment end setup camera
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.loadURDF("plane.urdf")
        self._p.resetDebugVisualizerCamera(3.7, -38, -69, [0, 0, 0])

    def create_sphere(center, radius):
        sphere = self._p.createVisualShape(
            pybullet.GEOM_SPHERE, radius=.1, rgbaColor=[1, 0, 0, 1])
        b = self._p.createMultiBody(baseVisualShapeIndex=sphere)
        self._p.resetBasePositionAndOrientation(  # position red sphere
            b, center + [0.], [0., 0., 0, 1])
        trajectory_spheres.append(b)

    def create_box(center, dim, orientation=np.identity(3)):
        box = self._p.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=[.5, .5, .5], rgbaColor=[0, .8, .8, 1])
