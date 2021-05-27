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
#                                        Jim Mainprice on Sunday March 14 2021


import socket
from socket import SHUT_RDWR
import sys
from pybewego.message_passing import *
from pyrieef.rendering.optimization import *
import traceback


class WorkspaceViewerServer(TrajectoryOptimizationViewer):
    """ Workspace display based on pyglet backend """

    def __init__(self, workspace, use_gl=True):
        TrajectoryOptimizationViewer.__init__(
            self,
            None,
            draw=False,
            draw_gradient=True,
            use_3d=False,
            use_gl=use_gl)

        # Create a TCP/IP socket
        self.address = ('127.0.0.1', 5555)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.address)

        # Listen for incoming connections
        self.socket.listen(1)

        # Store the active part of the trajectory
        self.q_init = None
        self.active_x = None

        self.init_viewer(workspace)

    def initialize_viewer(self, problem, trajectory):
        self.objective = problem
        self.viewer.background_matrix_eval = False
        self.viewer.save_images = True
        self.viewer.workspace_id += 1
        self.viewer.image_id = 0
        self.reset_objective()
        self.viewer.draw_ws_obstacles()
        self.q_init = trajectory.initial_configuration()
        self.active_shape = (self.objective.n * (self.objective.T + 1), )
        self.draw(trajectory)

    def run(self):
        stop = False
        while not stop:
            # Wait for a connection
            print('waiting for a connection...')
            connection, client_address = self.socket.accept()
            print('connection from', client_address)
            np.set_printoptions(linewidth=300)
            try:
                while True:
                    # Receive the data in small chunks and retransmit it
                    data = recv_msg(connection)
                    if data is not None:
                        # print("recieved data...")
                        data = data.decode("ascii")
                        # Check if the client is done.
                        if data == "end":
                            echo = "done"
                            connection.sendall(echo.encode("ascii"))
                            stop = True
                            break

                        # Deseralize data and check that all is ok
                        self.active_x = deserialize_array(data)
                        if self.active_x.shape == self.active_shape:
                            echo = "ackn"
                        else:
                            echo = "fail"
                        if not np.isnan(self.active_x).any() and (
                                np.abs(self.active_x).max() < 1e10):
                            # print(self.active_x)
                            trajectory = Trajectory(
                                self, q_init=self.q_init, x=self.active_x)
                            self.draw(trajectory)
                            self.viewer.draw_ws_circle(
                                self.objective.goal_manifold.radius,
                                self.objective.goal_manifold.origin,
                                color=(1, 0, 0))
                            # print("dist : ", np.linalg.norm(
                            #     self.objective.q_goal -
                            #     trajectory.final_configuration()))
                        connection.sendall(echo.encode("ascii"))

            except AssertionError:
                print("AssertionError")
                connection.sendall("fail".encode("ascii"))
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]
                print('An error occurred on line {} in statement {}'.format(
                    line, text))
                break

            finally:
                print("close connection.")
                # Clean up the connection
                # connection.close()
                # self.socket.shutdown(SHUT_RDWR)
                # self.socket.close()
                # self.viewer.gl.close()
