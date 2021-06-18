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
#                                               Jim Mainprice Fri June 18 2021

import demos_common_imports
import numpy as np
import time
from pybewego.pybullet_loader import PybulletRobot
from pybewego.kinematics import Kinematics
from pybewego import RobotOptimizer
import pybullet
import pybullet_data

trajectory_spheres = []
sphere_g = None
sphere_r = None


def create_sphere(p, radius):
    b = world.createMultiBody(baseVisualShapeIndex=sphere_g)
    world.resetBasePositionAndOrientation(  # position red sphere
        b, p + [0.], [0., 0., 0, 1])
    trajectory_spheres.append(b)

# load robot
robot = PybulletRobot("../data/r3_robot.urdf", with_gui=True)

# pybullet world and create floor
world = robot._p
world.setAdditionalSearchPath(pybullet_data.getDataPath())
world.loadURDF("plane.urdf")
world.resetDebugVisualizerCamera(3.7, -38, -69, [0, 0, 0])

# create red sphere
sphere_r = world.createVisualShape(pybullet.GEOM_SPHERE,
                                   radius=.1, rgbaColor=[1, 0, 0, 1])
sphere_g = world.createVisualShape(pybullet.GEOM_SPHERE,
                                   radius=.1, rgbaColor=[0, .8, .8, 1])
goal = world.createMultiBody(baseVisualShapeIndex=sphere_r)


# end-effector idx & arm dofs
eff_idx = 3
dofs = [0, 1, 2]

# Sample goal positions
np.random.seed(0)
positions = np.random.uniform(low=-3, high=3, size=(100, 2))


T = 30
n = len(dofs)
dt = 0.1
workspace = Workspace()
workspace_bounds = [-1, 1, -1, 1, -1, 1]
keypoints = [("link3", .01)]
kinematics = robot.create_robot(keypoints)

# for t in range(T):
#     create_sphere([0, 0, 0], .03)

problem = RobotMotionOptimization(
    kinematics, workspace, trajectory, dt, x_goal, workspace_bounds)
problem.verbose = False

p = CostFunctionParameters()
p.s_velocity_norm = 0
p.s_acceleration_norm = 10
p.s_obstacles = 1e+3
p.s_obstacle_alpha = 7
p.s_obstacle_gamma = 60
p.s_obstacle_margin = 0
p.s_obstacle_constraint = 1
p.s_terminal_potential = 1e+4

problem.initialize_objective(p)

for i, y_goal in enumerate(positions):

    i = 7
    y_goal = positions[i]
    print("i = ", i)

    world.resetBasePositionAndOrientation(  # position red sphere
        goal, y_goal.tolist() + [0.], [0., 0., 0, 1])

    q = robot.sample_config()[dofs]
    robot.set_and_update(q, dofs)

    q = robot.configuration()[dofs]
    y_0 = robot.position(eff_idx)[0:2]

    T = 100  # interpolation
    dist = 1

    dq = q

    while dist > .10 and np.linalg.norm(dq) > 1e-3:

        # get forward kinematics and jacobian
        robot.set_and_update(q, dofs)

        y = robot.position(eff_idx)[0:2]
        J = robot.jacobian_pos(eff_idx)[0:2, 6:9]

        dist = np.linalg.norm(y - y_goal)

        # print(J)
        # J_inv = np.linalg.pinv(J)
        J_inv = np.linalg.inv(J.T @ J + 1e-1 * np.eye(3)) @ J.T
        # J_inv = J.T @ np.linalg.inv(J @ J.T + 1e-1 * np.eye(2))
        # J_inv = J.T

        # print("J_inv : \n",J_inv)
        # print("J_inv_2 : \n",J_inv_2)

        dq = eta * J_inv @ (y - y_goal) / dist
        q = q - dq
        # q = q + J.T @ (y_i - y)

        # create_sphere(y.tolist(), .05)
        time.sleep(.01)

    for b in trajectory:
        world.removeBody(b)
    trajectory = []
    # print("sleep...")
    # time.sleep(5)
