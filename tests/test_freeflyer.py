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
#                                            Jim Mainprice on Sat Jun 11 2021

import os
import sys
import numpy as np
import conftest
from pybewego import create_freeflyer
from pyrieef.kinematics.robot import *
from numpy.testing import assert_allclose
import time


# directory = os.path.abspath(os.path.dirname(__file__))
# DATADIR = directory + "/../../pybullet_robots/data/"

def failing_test_freeflyer():
    """

    TODO: FIX THAT it sometimes fail (depending on the seed)

        - seed 0 works
        - seed 1 fails

    """
    np.random.seed(1) 
    s1 = Segment(p1=[1, 1], p2=[1, 7])
    s2 = Segment(p1=[1, 1], p2=[7, 1])
    nb_keypoints = 10
    radius = .7
    keypoints = create_keypoints(nb_keypoints, [s1, s2])
    # keypoints = [np.array([0, 0]), np.array([1, 1])]
    print(keypoints)
    radii = radius * np.ones(nb_keypoints)
    freeflyer1 = create_freeflyer("ff_test_2d", keypoints, radii.tolist())
    freeflyer2 = create_freeflyer_from_segments(nb_keypoints=nb_keypoints)
    configurations = np.random.random((1, 3))
    for q in configurations:
        for i in range(nb_keypoints):
            x1 = freeflyer1.keypoint_map(i)(q)
            print("{} -> {}".format(q, x1))
            x2 = freeflyer2.keypoint_map(i)(q)
            print("{} -> {}".format(q, x2))
            assert_allclose(x1, x2)


if __name__ == '__main__':
    failing_test_freeflyer()
