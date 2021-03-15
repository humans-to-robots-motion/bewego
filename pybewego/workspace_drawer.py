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

    def initialize_viewer(self, trajectory):
        self.viewer.background_matrix_eval = False
        self.viewer.save_images = True
        self.viewer.workspace_id += 1
        self.viewer.image_id = 0
        self.reset_objective()
        self.viewer.draw_ws_obstacles()
        self

    def run(self):

        while True:
            # Wait for a connection
        print('waiting for a connection')
        connection, client_address = sock.accept()

        try:
            print('connection from', client_address)
            # Receive the data in small chunks and retransmit it
            while True:
                data = connection.recv(1024)
                print("received ", str(data))
                if data:
                    print('sending data back to the client')
                    connection.sendall(data)
                else:
                    print('no more data from', client_address)
                    break

        finally:
            # Clean up the connection
            connection.close()


def setup_echo_tcp_server():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('127.0.0.1', 5555)
    print('starting up on {} port {}'.format(
        server_address[0], server_address[1]))
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    while True:
        # Wait for a connection
        print('waiting for a connection')
        connection, client_address = sock.accept()

        try:
            print('connection from', client_address)
            # Receive the data in small chunks and retransmit it
            while True:
                data = connection.recv(1024)
                print("received ", str(data))
                if data:
                    print('sending data back to the client')
                    connection.sendall(data)
                else:
                    print('no more data from', client_address)
                    break

        finally:
            # Clean up the connection
            connection.close()
