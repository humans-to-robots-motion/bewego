#!/usr/bin/env python

# Copyright (c) 2018, University of Stuttgart
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
#                                        Jim Mainprice on Sunday June 13 2018

import demos_common_imports
import numpy as np
from tqdm import tqdm

from pybewego.motion_optimization import MotionOptimization
from pybewego.motion_optimization import CostFunctionParameters

from pyrieef.optimization import algorithms
from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *

DRAW_MODE = "pyglet2d"  # None, pyglet2d, pyglet3d or matplotlib
VERBOSE = True
BOXES = True

nb_points = 40  # points for the grid on which to perform graph search.
np.random.seed(0)
sampling = sample_box_workspaces if BOXES else sample_circle_workspaces
for k, workspace in enumerate(tqdm([sampling(5) for i in range(100)])):

    trajectory = linear_interpolation_trajectory(
        q_init=np.zeros(2),
        q_goal=np.zeros(2),
        T=30)

    problem = MotionOptimization(
        workspace,
        trajectory,
        dt=0.01,
        q_goal=np.ones(2))

    problem.initialize_objective(CostFunctionParameters())

    objective = TrajectoryOptimizationViewer(
        problem,
        draw=DRAW_MODE is not None,
        draw_gradient=True,
        use_3d=DRAW_MODE == "pyglet3d",
        use_gl=DRAW_MODE == "pyglet2d")

    if DRAW_MODE is not None:
        objective.reset_objective()
        objective.viewer.save_images = True
        objective.viewer.workspace_id += 1
        objective.viewer.image_id = 0
        objective.viewer.draw_ws_obstacles()

    algorithms.newton_optimize_trajectory(
        objective, path, verbose=VERBOSE, maxiter=20)
