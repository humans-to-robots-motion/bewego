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
#                                    Jim Mainprice on Wednesday February 3 2021

from pybewego import MotionObjective
try:
    from pybewego import PlanarOptimizer
    from pybewego import RobotOptimizer
    WITH_IPOPT = True
except ImportError:
    WITH_IPOPT = False

from pyrieef.motion.trajectory import Trajectory
from pyrieef.geometry.workspace import *
from pyrieef.kinematics.robot import *

from scipy import optimize
import numpy as np


class CostFunctionParameters:

    """
    A cost function is parameterized by a small set of scalars
    """

    def __init__(self):
        self.s_velocity_norm = 1
        self.s_acceleration_norm = 1
        self.s_obstacles = 1
        self.s_obstacle_alpha = 10
        self.s_obstacle_gamma = 100
        self.s_obstacle_margin = 0
        self.s_obstacle_constraint = 0
        self.s_waypoint_constraint = 0
        self.s_terminal_potential = 1e+5

    def __str__(self):
        msg = str()
        msg += " - s_velocity_norm : {}\n".format(self.s_velocity_norm)
        msg += " - s_acceleration_norm : {}\n".format(self.s_acceleration_norm)
        msg += " - s_obstacles : {}\n".format(self.s_obstacles)
        msg += " - s_obstacle_alpha : {}\n".format(self.s_obstacle_alpha)
        msg += " - s_obstacle_margin : {}\n".format(self.s_obstacle_margin)
        msg += " - s_obstacle_constraint : {}\n".format(
            self.s_obstacle_constraint)
        msg += " - s_terminal_potential : {}\n".format(
            self.s_terminal_potential)
        return msg


class MotionOptimization:

    """
    Motion optimization class
    that can draw the inner optimization quantities

    Parameters
    ----------
        workspace: Workspace
            contains the geometry of the problem
        trajectory: Trajectory
            contains the initial trajectory to optimizer
        dt: float
            time discretization
        q_goal: array
            goal configuration
    """

    def __init__(self, workspace, trajectory, dt, q_goal):

        self.dt = dt
        self.n = trajectory.n()
        self.T = trajectory.T()
        self.q_init = trajectory.initial_configuration()
        self.q_goal = q_goal
        self.q_waypoint = None
        self.workspace = workspace
        self.trajectory = trajectory
        self.verbose = False
        self.problem = None

    def _problem(self):
        """
        this function can be derived to use different
        versions of the motion optimization objective 
        """
        return MotionObjective(self.T, self.dt, self.n)

    def _initialize_problem(self):
        """
        Initialize the motion optimization problem
        based on the current version of the workspace
        """
        self.problem = self._problem()

        # Add workspace obstacles
        for o in self.workspace.obstacles:
            if hasattr(o, '_is_circle'):
                self.problem.add_sphere(o.origin, o.radius)
            elif hasattr(o, '_is_box'):
                self.problem.add_box(o.origin, o.dim)
            else:
                print("Shape {} not supported by bewego".format(type(o)))

    def initialize_objective(self, scalars):
        """
        Initialize the motion optimization problem
        based on the scalars given as input

        Parameters
        ----------
        scalars :
            CostFunctionParameters contains scalars that define the
            motion optimization objective
        """
        self._initialize_problem()

        # Terms
        if scalars.s_velocity_norm > 0:
            self.problem.add_smoothness_terms(
                1, scalars.s_velocity_norm)

        if scalars.s_acceleration_norm > 0:
            self.problem.add_smoothness_terms(
                2, scalars.s_acceleration_norm)

        if scalars.s_obstacles > 0:
            self.problem.set_sdf_gamma(scalars.s_obstacle_gamma)
            self.problem.set_sdf_margin(scalars.s_obstacle_margin)
            self.problem.add_obstacle_terms(
                scalars.s_obstacles,
                scalars.s_obstacle_alpha)

        if scalars.s_terminal_potential > 0:
            self.problem.add_terminal_potential_terms(
                self.q_goal, scalars.s_terminal_potential)

        # Create objective functions
        self.objective = self.problem.objective(self.q_init)
        self.obstacle_potential = self.problem.obstacle_potential()  # TODO

    def optimize(self,
                 scalars,
                 nb_steps=100,
                 optimizer="newton"):

        self.initialize_objective(scalars)
        xi = self.trajectory.active_segment()

        if optimizer is "newton":
            res = optimize.minimize(
                x0=np.array(xi),
                method='Newton-CG',
                fun=self.objective.forward,
                jac=self.objective.gradient,
                hess=self.objective.hessian,
                options={'maxiter': nb_steps, 'disp': self.verbose}
            )
            self.trajectory.active_segment()[:] = res.x
            gradient = res.jac
            delta = res.jac
            dist = np.linalg.norm(
                self.trajectory.final_configuration() - self.q_goal)
            if self.verbose:
                print(("gradient norm : ", np.linalg.norm(res.jac)))
        else:
            raise ValueError

        return [dist < 1.e-3, trajectory, gradient, delta]


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if WITH_IPOPT:  # only define class if bewego is compiled with IPOPT

    class IpoptMotionOptimization(MotionOptimization):

        """
        Navigation Optimization

            This class allows plan 2D trajectories using IPOPT

        Parameters
        ----------
            workspace : 
                Workspace object
            trajectory : 
                Trajectory object it is theinitial trajectory
            dt : 
                Float, time between each configuration in the trajectory
            q_goal :
                np.array, configuration at the goal
            bounds :
                np.array, [x_min, x_max, y_min, y_max]
        """

        def __init__(self, workspace, trajectory, dt, q_goal,
                     bounds=[0., 1., 0., 1.]):

            MotionOptimization.__init__(
                self, workspace, trajectory, dt, q_goal)

            self.bounds = bounds
            self.with_goal_constraint = False
            self.with_goal_manifold_constraint = True
            self.with_smooth_obstale_constraint = True
            self.with_waypoint_constraint = True

        def initialize_objective(self, scalars):
            """
            Initialize the motion optimization problem
            based on the scalars given as input
            These are common terms shared across the different problems

            Parameters
            ----------
            scalars :
                CostFunctionParameters contains scalars that define the
                motion optimization objective
            """
            self._initialize_problem()

            # Objective Terms -------------------------------------------------

            if scalars.s_velocity_norm > 0:
                print("-- add velocity norm ({})".format(
                    scalars.s_velocity_norm))
                self.problem.add_smoothness_terms(
                    1, scalars.s_velocity_norm)

            if scalars.s_acceleration_norm > 0:
                print("-- add acceleration norm ({})".format(
                    scalars.s_acceleration_norm))
                self.problem.add_smoothness_terms(
                    2, scalars.s_acceleration_norm)

            if scalars.s_obstacles > 0:
                print("-- add obstaces term ({})".format(
                    scalars.s_obstacles))
                self.problem.set_sdf_gamma(scalars.s_obstacle_gamma)
                self.problem.set_sdf_margin(scalars.s_obstacle_margin)
                self.problem.add_obstacle_terms(
                    scalars.s_obstacles,
                    scalars.s_obstacle_alpha)

            if ((not self.with_goal_constraint) and
                    (scalars.s_terminal_potential > 0)):
                print("-- add terminal dist ({})".format(
                    scalars.s_terminal_potential))
                self.problem.add_terminal_potential_terms(
                    self.q_goal, scalars.s_terminal_potential)

            # Constraints Terms -----------------------------------------------

            if self.with_goal_constraint and scalars.s_terminal_potential > 0:
                print("-- add goal constraint ({})".format(
                    scalars.s_terminal_potential))
                self.problem.add_goal_constraint(
                    self.q_goal, scalars.s_terminal_potential)

            if scalars.s_obstacle_constraint > 0:
                print("-- add keypoints surface constraint ({})".format(
                    scalars.s_obstacle_constraint))
                if self.with_smooth_obstale_constraint:
                    self.problem.add_smooth_keypoints_surface_constraints(
                        scalars.s_obstacle_constraint)
                else:
                    self.problem.add_keypoints_surface_constraints(
                        scalars.s_obstacle_margin,
                        scalars.s_obstacle_constraint)

        def create_objective_functions(self):
            self.objective = self.problem.objective(self.q_init)
            self.obstacle_potential = self.problem.obstacle_potential()  # TODO
            self.problem.set_trajectory_publisher(False, 100000)

    class NavigationOptimization(IpoptMotionOptimization):

        """
        Navigation Optimization

            This class allows plan 2D trajectories using IPOPT

        Parameters
        ----------
            workspace : 
                Workspace object
            trajectory : 
                Trajectory object it is theinitial trajectory
            dt : 
                Float, time between each configuration in the trajectory
            q_goal :
                np.array, configuration at the goal
            bounds :
                np.array, [x_min, x_max, y_min, y_max]
        """

        def __init__(self, workspace, trajectory, dt, q_goal, goal_radius=0.1,
                     q_waypoint=None,
                     bounds=[0., 1., 0., 1.]):
            IpoptMotionOptimization.__init__(
                self, workspace, trajectory, dt, q_goal, bounds)
            assert len(q_goal) == 2
            assert len(bounds) == 4
            self.bounds = bounds
            self.with_goal_constraint = False
            self.with_goal_manifold_constraint = True
            self.with_smooth_obstale_constraint = True
            self.with_waypoint_constraint = True
            self.q_waypoint = q_waypoint
            self.goal_manifold = Circle(origin=q_goal, radius=goal_radius)

        def _problem(self):
            """ This version of the problem uses constraints """
            return PlanarOptimizer(self.T, self.dt, self.bounds)

        def initialize_objective(self, scalars):
            """
            Initialize the motion optimization problem
            based on the scalars given as input

            Parameters
            ----------
            scalars :
                CostFunctionParameters contains scalars that define the
                motion optimization objective
            """
            IpoptMotionOptimization.initialize_objective(self, scalars)

            # Mainfold and waypoints terms

            if (self.with_goal_manifold_constraint and
                    scalars.s_terminal_potential > 0):
                print("-- add goal manifold constraint ({})".format(
                    scalars.s_terminal_potential))
                self.problem.add_goal_manifold_constraint(
                    self.goal_manifold.origin,
                    self.goal_manifold.radius,
                    scalars.s_terminal_potential)

            if (self.with_waypoint_constraint and
                    scalars.s_waypoint_constraint > 0 and
                    self.q_waypoint is not None):
                print("-- add waypoint constraint ({})".format(
                    scalars.s_waypoint_constraint))
                self.problem.add_waypoint_constraint(
                    self.q_waypoint, 15, scalars.s_waypoint_constraint)

            self.create_objective_functions()

        def optimize(self,
                     scalars,
                     ipopt_options={}):
            self.initialize_objective(scalars)

            res = self.problem.optimize(
                self.trajectory.x(),
                self.q_goal,
                ipopt_options
            )
            self.trajectory.active_segment()[:] = res.x
            dist = np.linalg.norm(
                self.trajectory.final_configuration() - self.q_goal)
            if self.verbose:
                print(("gradient norm : ", np.linalg.norm(res.jac)))
            return [dist < 1.e-3, self.trajectory]

    class FreeflyerOptimization(NavigationOptimization):

        """
        Freeflyer Optimization

            This class allows plan 2D trajectories using IPOPT

        Parameters
        ----------
            workspace : 
                Workspace object
            trajectory : 
                Trajectory object it is theinitial trajectory
            dt : 
                Float, time between each configuration in the trajectory
            q_goal :
                np.array, configuration at the goal
            bounds :
                np.array, [x_min, x_max, y_min, y_max]
        """

        def __init__(self, workspace, trajectory, dt, q_goal, goal_radius=0.1,
                     q_waypoint=None,
                     bounds=[0., 1., 0., 1.]):
            NavigationOptimization.__init__(
                workspace, trajectory, dt, q_goal,
                goal_radius=goal_radius,
                q_waypoint=q_waypoint,
                bounds=bounds)

        def _problem(self):
            """ This version of the problem uses constraints """
            freeflyer = create_robot_from_file()
            return FreeflyerOptimizer(
                3, self.T, self.dt, self.bounds,
                "freeflyer_2d")

    class RobotMotionOptimization(IpoptMotionOptimization):

        """
        Robot Optimization

            This class allows plan N dimensional trajectories using IPOPT

        Parameters
        ----------
            robot :
                A bewego robot (i.e., defines task maps)
            workspace : 
                Workspace object
            trajectory : 
                Trajectory object it is theinitial trajectory
            dt : 
                Float, time between each configuration in the trajectory
            q_goal :
                np.array, configuration at the goal
            bounds :
                np.array, [x_min, x_max, y_min, y_max]
        """

        def __init__(self, robot, workspace, trajectory, dt, x_goal,
                     bounds=[0., 1., 0., 1., 0., 1.]):
            IpoptMotionOptimization.__init__(
                workspace, trajectory, dt, x_goal,
                bounds=bounds)

            assert robot.keypoint_map(0).input_dimension() == self.n
            assert len(bounds) == 6

            self.robot = robot

        def _problem(self):
            """ This version of the problem uses constraints """
            return RobotOptimizer(
                self.n, self.T, self.dt, self.bounds, self.robot)

        def initialize_objective(self, scalars):
            """
            Initialize the motion optimization problem
            based on the scalars given as input

            Parameters
            ----------
            scalars :
                CostFunctionParameters contains scalars that define the
                motion optimization objective
            """
            IpoptMotionOptimization.initialize_objective(self, scalars)
            self.create_objective_functions()
