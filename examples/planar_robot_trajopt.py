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
from pybewego.pybullet_world import PybulletRobot
from pybewego.kinematics import Kinematics
from pybewego import RobotOptimizer
import pybullet
import pybullet_data

trajectory_spheres = []
sphere_g = None
sphere_r = None

# specify the workspace
workspace = Workspace()
workspace.obstacles.append(
    OrientedBox(dim=[.2, .5, .1], orienation=np.identity(3)))
workspace_bounds = [-1, 1, -1, 1, -1, 1]

# load robot
robot = PybulletRobot("../data/r3_robot.urdf", with_gui=True)

# pybullet world and create floor
world = PybulletWorld(robot, workspace)

# end-effector idx & arm dofs
eff_idx = 3
dofs = [0, 1, 2]

# Sample goal positions
np.random.seed(0)
positions = np.random.uniform(low=-3, high=3, size=(100, 2))

# Trajectory parameters
T = 30
n = len(dofs)
dt = 0.1

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
