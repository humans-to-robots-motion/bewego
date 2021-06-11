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
from pybewego import *
from pyrieef.kinematics.robot import *
from numpy.testing import assert_allclose
import time


# directory = os.path.abspath(os.path.dirname(__file__))
# DATADIR = directory + "/../../pybullet_robots/data/"

def test_freeflyer():
    segment1 = [5.5, 0.5, 0, 0.5, 0.5, 0]
    segment2 = [0.5, 0.5, 0, 0.5, 5.5, 0]
    keypoints = create_keypoints(10, segment1, segment2)
    freeflyer = create_freeflyer("ff_test_2d", )  # TODO
