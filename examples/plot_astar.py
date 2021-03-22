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
import os
import sys
sys.path.insert(0, "/Users/jmainpri/Dropbox/Work/workspace/dstar/python")

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
from pydstar import Dstar2D

show_result = True
radius = .1
nb_points = 40
average_cost = False


def trajectory(pixel_map, path):
    trajectory = [None] * len(path)
    for i, p in enumerate(path):
        trajectory[i] = pixel_map.grid_to_world(np.array(p))
    return trajectory


workspace = Workspace()
workspace.obstacles.append(Circle(np.array([0.1, 0.1]), radius))
workspace.obstacles.append(Circle(np.array([-.1, 0.1]), radius))
phi = CostGridPotential2D(SignedDistanceWorkspaceMap(workspace), 10., .1, 10.)
costmap = phi(workspace.box.stacked_meshgrid(nb_points))
print(costmap)

converter = CostmapToSparseGraph(costmap, average_cost)
converter.integral_cost = True
graph = converter.convert()
if average_cost:
    assert check_symmetric(graph)
# predecessors = shortest_paths(graph)
pixel_map = workspace.pixel_map(nb_points)
np.random.seed(1)
for i in range(100):
    s_w = sample_collision_free(workspace)
    t_w = sample_collision_free(workspace)
    s = pixel_map.world_to_grid(s_w)
    t = pixel_map.world_to_grid(t_w)
    if s[0] == 0 or s[1] == 0:
        continue
    if t[0] == 0 or t[1] == 0:
        continue

    try:
        time_0 = time.time()
        print("planning (1)...")
        path1 = converter.dijkstra_on_map(costmap, s[0], s[1], t[0], t[1])
    except:
        continue
    print("1) took t : {} sec.".format(time.time() - time_0))
    # try:

    time_0 = time.time()
    print("planning (2)...")
    print(costmap.shape)
    astar = AStarGrid()
    astar.init_grid(1. / nb_points, [0, 1, 0, 1])
    astar.set_costs(costmap)
    assert astar.solve(s, t)
    path2 = astar.path()
    print("2) took t : {} sec.".format(time.time() - time_0))

    time_0 = time.time()
    print("planning (3)...")
    print(costmap.shape)
    dstar = Dstar2D()
    assert dstar.solve(s, t, costmap)
    path3 = dstar.path().T
    print("3) took t : {} sec.".format(time.time() - time_0))

    # time_0 = time.time()
    # print("planning (4)...")
    # print(costmap.shape)
    # print("s : ", s)
    # print("t : ", t)
    # viter = ValueIteration()
    # path4 = viter.solve(s, t, costmap)
    # print("4) took t : {} sec.".format(time.time() - time_0))

    if show_result:

        viewer = render.WorkspaceDrawer(
            rows=1, cols=3, workspace=workspace, wait_for_keyboard=True)

        viewer.set_drawing_axis(0)
        # viewer.draw_ws_background(phi, nb_points, interpolate="none")
        viewer.draw_ws_img(costmap.T, interpolate="none")
        viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory(pixel_map, path1))
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)

        viewer.set_drawing_axis(1)
        viewer.draw_ws_background(phi, nb_points, interpolate="none")
        viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory(pixel_map, path2), "b")
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)

        viewer.set_drawing_axis(2)
        viewer.draw_ws_background(phi, nb_points, interpolate="none")
        viewer.draw_ws_obstacles()
        viewer.draw_ws_line(trajectory(pixel_map, path3), "g")
        viewer.draw_ws_point(s_w)
        viewer.draw_ws_point(t_w)

        # viewer.set_drawing_axis(3)
        # viewer.draw_ws_background(phi, nb_points, interpolate="none")
        # viewer.draw_ws_obstacles()
        # viewer.draw_ws_line(trajectory(pixel_map, path4), "r")
        # viewer.draw_ws_point(s_w)
        # viewer.draw_ws_point(t_w)

        viewer.show_once()

print(path)
