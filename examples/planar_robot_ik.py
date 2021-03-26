#!/usr/bin/env python

# Copyright (c) 2019, University of Stuttgart
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
#                                       Jim Mainprice on Thursday June 13 2019

import demos_common_imports
import numpy as np
import time
from pybewego.pybullet_loader import PybulletRobot
import pybullet
import pybullet_data

# load robot
robot = PybulletRobot("../data/r3_robot.urdf", with_gui=True)

# pybullet world and create floor
world = robot._p
world.setAdditionalSearchPath(pybullet_data.getDataPath())
world.loadURDF("plane.urdf")
world.resetDebugVisualizerCamera(3.7, -38, -69, [0, 0, 0])

# create red sphere
sphere = world.createVisualShape(pybullet.GEOM_SPHERE,
                                 radius=.1, rgbaColor=[1, 0, 0, 1])
goal = world.createMultiBody(baseVisualShapeIndex=sphere)

# end-effector idx & arm dofs
eff_idx = 3
dofs = [0, 1, 2]

# Sample goal positions
np.random.seed(0)
positions = np.random.uniform(low=-3, high=3, size=(100, 2))


for i, y_goal in enumerate(positions):

    i = 8
    y_goal = positions[i]
    print("i = ", i)

    world.resetBasePositionAndOrientation(  # position red sphere
        goal, y_goal.tolist() + [0.], [0., 0., 0, 1])

    q = robot.sample_config()[dofs]
    robot.set_and_update(q, dofs)

    q = robot.get_configuration()[dofs]
    y_0 = robot.get_position(eff_idx)[0:2]

    T = 100  # interpolation
    for t in range(T):

        # get forward kinematics and jacobian
        y = robot.get_position(eff_idx)[0:2]
        J = robot.get_jacobian(eff_idx)[0:2, 6:9]

        y_i = y_0 + (t / float(T)) * (y_goal - y_0)

        J_inv = np.linalg.pinv(J)
        # J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-1 * np.eye(2))

        q = q + J_inv @ (y_i - y)
        # q = q + J.T @ (y_i - y)

        robot.set_and_update(q, dofs)
        time.sleep(.01)
