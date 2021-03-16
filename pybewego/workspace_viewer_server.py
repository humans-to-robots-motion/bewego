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
#                                        Jim Mainprice on Sunday March 14 2021


import socket
import sys
from pybewego.message_passing import *
from pyrieef.rendering.optimization import *


class WorkspaceViewerServer(TrajectoryOptimizationViewer):
    """ Workspace display based on pyglet backend """

    def __init__(self, problem):
        TrajectoryOptimizationViewer.__init__(
            self,
            problem,
            draw=True,
            draw_gradient=True,
            use_3d=False,
            use_gl=True)

        # Create a TCP/IP socket
        self.address = ('127.0.0.1', 5555)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(self.address)

        # Listen for incoming connections
        self.socket.listen(1)

        # Store the active part of the trajectory
        self.active_x = None
        self.active_shape = (3, 10)
        # self.active_shape = (self.objective.n * (self.objective.T + 1), )

    def initialize_viewer(self, trajectory):
        self.viewer.background_matrix_eval = False
        self.viewer.save_images = True
        self.viewer.workspace_id += 1
        self.viewer.image_id = 0
        self.reset_objective()
        self.viewer.draw_ws_obstacles()
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
                    data = connection.recv(1024).decode("ascii")
                    if data:
                        # Check if the client is done.
                        if data == "end":
                            echo = "done"
                            print("send back echo : ", echo)
                            connection.sendall(echo.encode("ascii"))
                            stop = True
                            break

                        # Deseralize data and check that all is ok
                        self.active_x = deserialize_array(data)
                        if self.active_x.shape == self.active_shape:
                            echo = "ackn"
                        else:
                            echo = "fail"
                        print(self.active_x)
                        # print(self.active_x.shape)
                        print("send back echo : ", echo)
                        connection.sendall(echo.encode("ascii"))

            finally:
                print("close connection.")
                # Clean up the connection
                connection.close()
