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
#                                            Jim Mainprice on Sat Jun 6 2020

import os
import sys
import numpy as np
driectory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, driectory)
sys.path.insert(0, driectory + os.sep + "../python")
from pybewego import AStarGrid


def test_set_env_size():
    astar = AStarGrid()

    v1 = [0, 2, 0, 2]
    astar.set_env_size(v1)
    v2 = astar.env_size()
    assert np.linalg.norm((np.array(v1) - np.array(v2))) < 1e-10

    v3 = [0, 1, 0, 1]
    astar.set_env_size(v3)
    v2 = astar.env_size()
    assert np.linalg.norm((np.array(v3) - np.array(v2))) < 1e-10


def test_set_pace():
    astar = AStarGrid()

    v1 = .1
    astar.set_pace(v1)
    v2 = astar.pace()
    d = abs(v1 - v2)
    print(d)
    assert d < 1e-10

    v3 = .2
    astar.set_pace(v3)
    v2 = astar.pace()
    d = abs(v2 - v3)
    print(d)
    assert d < 1e-10


test_set_env_size()
test_set_pace()
