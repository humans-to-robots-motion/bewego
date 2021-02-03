#!/usr/bin/env python

# Copyright (c) 2021, University of Stuttgart
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
#                                    Jim Mainprice on Wednesday February 3 2021


import pybullet_utils.bullet_client as bc
import pybullet
from kinematic_structures import ScalarBounds
from kinematic_structures import RigidBody
from kinematic_structures import transform
import numpy as np


def print_joint_info(info):
    print("Joint info: " + str(info))
    print(" - joint name : ", info[1])      # name of the joint in URDF
    print(" - type : ", info[2])            # p.JOINT_REVOLUTE or p.PRISMATIC
    print(" - limit (l) : ", info[8])       # lower limit
    print(" - limit (h) : ", info[9])       # higher limit
    print(" - body name : ", info[12])
    print(" - axis : ", info[13])           # joint axis
    print(" - origin (p) : ", info[14])     # position 3d
    print(" - origin (r) : ", info[15])     # quaternion 4d


def CreateRobotFromURDF(filename):

    # connect to pybullet
    # new direct client, GUI for graphic
    p = bc.BulletClient(connection_mode=pybullet.DIRECT)

    rigid_bodies = []

    # load robot
    robot = p.loadURDF(filename)
    njoints = p.getNumJoints(robot)
    for i in range(njoints):
        info = p.getJointInfo(robot, i)
        # print_joint_info(info)
        rigid_body = RigidBody()
        rigid_body.name = info[12]
        rigid_body.joint_bounds = ScalarBounds(info[8], info[9])
        rigid_body.joint_name = info[1]
        rigid_body.joint_axis_in_local = np.asarray(info[13])
        rigid_body.frame_in_local = transform(
            np.asarray(info[14]),
            np.asarray(info[15])
        )
        print(rigid_body)
        rigid_bodies.append(rigid_bodies)

    print("number joints: " + str(njoints))
