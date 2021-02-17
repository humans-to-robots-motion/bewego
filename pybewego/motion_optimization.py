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

from pybewego import MotionObjective
from pyrieef.motion.trajectory import Trajectory
from pyrieef.geometry.workspace import *

class MotionOptimization:

    """ Motion optimization class
        that can draw the inner optimization quantities """

    def __init__(self, workspace, trajectory, dt, q_goal ):

        self.dt = dt
        self.n = trajectory.n()
        self.T = trajectory.T()
        self.q_init = trajectory.initial_configuration()
        self.q_goal = q_goal
        self.workspace = workspace
        self.problem = None

    def initialize_objective(self):

        self.problem = MotionObjective(self.T, self.dt, self.n)

        # Add workspace obstacles
        for o in self.workspace.obstacles:
            if isinstance(o, Circle):
                self.problem.add_sphere(o.origin, o.radius)
            elif isinstance(o, Box):
                self.problem.add_box(o.origin, o.dim)
            else:
                print("Shape not supported by bewego")

        # Terms
        self.problem.add_smoothness_terms(1, scalar_v)
        self.problem.add_smoothness_terms(2, scalar_a)
        self.problem.add_obstacle_terms(scalar_o, alpha, margin)
        self.problem.add_terminal_potential_terms(self.q_goal, scalar_g)

        # Create objective functions
        self.objective = self.problem.objective(q_goal)
        self.obstacle_potential = self.problem.obstacle_potential() # TODO


    def optimize(self,
                 q_init,
                 nb_steps=100,
                 trajectory=None,
                 optimizer="natural_gradient"):

        if trajectory is None:
            trajectory = linear_interpolation_trajectory(
                q_init, self.q_goal, self.T)

        xi = trajectory.active_segment()

        if optimizer is "newton":
            res = optimize.minimize(
                x0=np.array(xi),
                method='Newton-CG',
                fun=self.objective.forward,
                jac=self.objective.gradient,
                hess=self.objective.hessian,
                options={'maxiter': nb_steps, 'disp': self.verbose}
            )
            trajectory.active_segment()[:] = res.x
            gradient = res.jac
            delta = res.jac
            dist = np.linalg.norm(
                trajectory.final_configuration() - self.q_goal)
            if self.verbose:
                print(("gradient norm : ", np.linalg.norm(res.jac)))
        else:
            raise ValueError

        return [dist < 1.e-3, trajectory, gradient, delta]
