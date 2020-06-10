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

import os
import sys
driectory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, driectory)
sys.path.insert(0, driectory + os.sep + "../python")
sys.path.insert(0, driectory + os.sep + "../../pyrieef")

import numpy as np
from numpy.testing import assert_allclose
from pyrieef.graph.shortest_path import *
from pyrieef.geometry.workspace import *
from pyrieef.motion.cost_terms import *
import pyrieef.rendering.workspace_renderer as render
from utils import timer
import time
from pybewego import AStarGrid
from pybewego import ValueIteration

show_result = True
radius = .1
nb_points = 10
average_cost = False


def trajectory(pixel_map, path):
    trajectory = [None] * len(path)
    for i, p in enumerate(path):
        trajectory[i] = pixel_map.grid_to_world(np.array(p))
    return trajectory


workspace = Workspace()
# workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
# workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
phi = CostGridPotential2D(SignedDistanceWorkspaceMap(workspace), 10., .1, 10.)
costmap = phi(workspace.box.stacked_meshgrid(nb_points))
# print(costmap)

pixel_map = workspace.pixel_map(nb_points)
np.random.seed(2)
for i in range(100):
    s_w = sample_collision_free(workspace)
    t_w = sample_collision_free(workspace)
    s = pixel_map.world_to_grid(s_w)
    t = pixel_map.world_to_grid(t_w)
    if s[0] == 0 or s[1] == 0:
        continue
    if t[0] == 0 or t[1] == 0:
        continue

    time_0 = time.time()
    print("planning (4)...")
    print(costmap.shape)
    print("s : ", s)
    print("t : ", t)
    viter = ValueIteration()
    viter.set_max_iterations(100)
    C = np.ones(costmap.shape)
    C[t[0], t[1]] = 0
    Vt = viter.run(C, t)
    V = np.ones(costmap.shape) * Vt.min()
    V[1:-1, 1:-1] = Vt[1:-1, 1:-1]
    # path = viter.solve(s, t, costmap)
    print("4) took t : {} sec.".format(time.time() - time_0))

    if show_result:

        viewer = render.WorkspaceDrawer(
            rows=1, cols=1, workspace=workspace, wait_for_keyboard=True)

        viewer.set_drawing_axis(0)
        viewer.draw_ws_img(V, interpolate="none")
        viewer.draw_ws_obstacles()
        # viewer.draw_ws_line(trajectory(pixel_map, path))
        # viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w, "r")

        # viewer.set_drawing_axis(1)
        # viewer.draw_ws_img(C, interpolate="none")
        # viewer.draw_ws_obstacles()
        # # viewer.draw_ws_point(s_w)
        # viewer.draw_ws_point(t_w, "r")

        viewer.show_once()

print(path)
