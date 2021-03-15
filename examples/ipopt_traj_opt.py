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
from multiprocessing import Process

from pybewego.motion_optimization import NavigationOptimization
from pybewego.motion_optimization import CostFunctionParameters

from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *
import pyrieef.learning.demonstrations as demonstrations
from pyrieef.graph.shortest_path import *

DRAW_MODE = "pyglet2d"  # None, pyglet2d, pyglet3d or matplotlib
VERBOSE = True
BOXES = True


def run_optimizer(problem, p, options):
    print(p)
    problem.optimize(p, options)


nb_points = 40  # points for the grid on which to perform graph search.
grid = np.ones((nb_points, nb_points))
graph = CostmapToSparseGraph(grid, average_cost=False)
graph.convert()

np.random.seed(0)
sampling = sample_box_workspaces if BOXES else sample_circle_workspaces
for k, workspace in enumerate(tqdm([sampling(5) for i in range(100)])):

    trajectory = demonstrations.graph_search_path(
        graph, workspace, nb_points)
    if trajectory is None:
        continue

    problem = NavigationOptimization(
        workspace,
        trajectory,
        dt=0.01,
        q_goal=trajectory.final_configuration(),
        bounds=workspace.box.box_extent())
    problem.verbose = True

    p = CostFunctionParameters()
    p.s_velocity_norm = 0
    p.s_acceleration_norm = 1e-5
    p.s_obstacles = 0
    p.s_obstacle_alpha = 0
    p.s_obstacle_scaling = 0
    p.s_terminal_potential = 1

    options = {}
    options["tol"] = 1e-2
    options["acceptable_tol"] = 5e-3
    options["acceptable_constr_viol_tol"] = 5e-1
    options["max_cpu_time"] = 30
    options["constr_viol_tol"] = 5e-2
    options["max_iter"] = 200

    p = Process(target=run_optimizer, args=(problem, p, options))
    p.start()
    p.join()
