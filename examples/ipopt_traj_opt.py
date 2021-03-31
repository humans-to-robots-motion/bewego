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
from pybewego.workspace_viewer_server import WorkspaceViewerServer

from pyrieef.rendering.optimization import TrajectoryOptimizationViewer
from pyrieef.geometry.workspace import *
from pyrieef.motion.trajectory import *
from pyrieef.utils.collision_checking import *
import pyrieef.learning.demonstrations as demonstrations
from pyrieef.graph.shortest_path import *

import time

VERBOSE = False
BOXES = False
DRAW_MODE = "pyglet2d"  # None, pyglet2d, pyglet3d or matplotlib
NB_POINTS = 40          # points for the grid on which to perform graph search.
NB_PROBLEMS = 100       # problems to evaluate
TRAJ_LENGTH = 40

viewer = WorkspaceViewerServer(Workspace())
grid = np.ones((NB_POINTS, NB_POINTS))
graph = CostmapToSparseGraph(grid, average_cost=False)
graph.convert()
np.random.seed(0)
sampling = sample_box_workspaces if BOXES else sample_circle_workspaces
workspaces = [sampling(5) for i in range(NB_PROBLEMS)]

for k, workspace in enumerate(tqdm(workspaces)):
    print("K = ", k)
    # workspace = workspaces[9]
    np.random.seed(0)
    trajectory = demonstrations.graph_search_path(
        graph, workspace, NB_POINTS)
    if trajectory is None:
        continue

    problem = NavigationOptimization(
        workspace,
        resample(trajectory, TRAJ_LENGTH),
        # trajectory,
        dt=0.3 / float(TRAJ_LENGTH),
        q_goal=trajectory.final_configuration() + .05 * np.random.rand(2),
        q_waypoint=np.array([.5, .5]),
        bounds=workspace.box.box_extent())
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
    p.s_waypoint_constraint = 0
    problem.initialize_objective(p)

    # Initialize the viewer with objective function etc.
    viewer.initialize_viewer(problem, problem.trajectory)

    options = {}
    options["tol"] = 1e-2
    options["acceptable_tol"] = 5e-3
    options["acceptable_constr_viol_tol"] = 5e-1
    options["constr_viol_tol"] = 5e-2
    options["max_iter"] = 200
    # options["bound_relax_factor"] = 0
    options["obj_scaling_factor"] = 1e+2

    p = Process(target=problem.optimize, args=(p, options))
    p.start()
    print("run viewer...")
    t0 = time.time()
    viewer.run()
    p.join()
    print("time : ", time.time() - t0)

viewer.viewer.gl.close()
