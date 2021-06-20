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
import pybullet_data
from scipy.spatial.transform import Rotation
import numpy as np
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

    def add_sphere(self, center, radius,
                   color=[1, 0, 0, 1]):
        """
        Creates a sphere

        Attributes
        ----------
        center : numpy array
            3D vector in world coordinates
        radius : Float
            size of the sphere
        color : array of Float
            rgba vector (Red, Green, Blue, A)
        """
        sphere = self._p.createVisualShape(
            pybullet.GEOM_SPHERE,
            radius=radius, rgbaColor=color)

        # pose the body
        b = self._p.createMultiBody(baseVisualShapeIndex=sphere)
        self._p.resetBasePositionAndOrientation(  # position red sphere
            b, center, [0., 0., 0, 1])

    def add_box(self, center, extents, orientation=np.identity(3),
                color=[1, 0, 0, 1]):
        """
        Creates a box

        Attributes
        ----------
        center : numpy array
            3D vector in world coordinates
        extents : array of Float
            3D vector of dimensions
        color : array of Float
            rgba vector (Red, Green, Blue, A)
        """
        box = self._p.createVisualShape(
            pybullet.GEOM_BOX,
            halfExtents=np.asarray(extents) / 2., rgbaColor=color)

        # pose the body
        quaterinon = Rotation.from_matrix(orientation).as_quat()
        print(quaterinon)
        b = self._p.createMultiBody(baseVisualShapeIndex=box)
        self._p.resetBasePositionAndOrientation(  # position red box
            b, center, quaterinon)

    def add_workspace_obstacles(self, workspace, color=[.3, .3, .3, 1]):
        """
        Import workspace obstacles as pybullet bodies

        Attributes
        ----------
        workspace
        """
        for o in workspace.obstacles:
            if hasattr(o, '_is_circle'):
                self.add_sphere(o.origin, o.radius, color)
            elif hasattr(o, '_is_box'):
                self.add_box(o.origin, o.dim, np.identity(3), color)
            elif hasattr(o, '_is_oriented_box'):
                self.add_box(o.origin, o.dim, o.orientation, color)
            else:
                print("Shape {} not supported by bewego".format(type(o)))
