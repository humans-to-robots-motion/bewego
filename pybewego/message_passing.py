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
import numpy as np


def serialize_array(arr):
    """
    Serializes an array
    """
    assert len(arr.shape) <= 2
    nrows = arr.shape[0]
    ncols = 1 if len(arr.shape) < 2 else arr.shape[1]
    txt = str()

    # Vector case
    if len(arr.shape) < 2:
        txt += "vector\n"
        txt += "rows:" + str(nrows) + "\n"
        txt += "cols:" + str(ncols) + "\n"
        txt += np.array2string(
            arr.flatten(),
            separator=",")[1:-1].replace("\n", "")

    # Matrix case
    else:
        txt += "matrix\n"
        txt += "rows:" + str(nrows) + "\n"
        txt += "cols:" + str(ncols) + "\n"
        for row in arr:
            txt += np.array2string(
                row.flatten(),
                separator=",")[1:-1].replace("\n", "") + "\n"
        txt = txt.rstrip("\n")
    return txt


def deserialize_array(txt):
    """
    Deserializes an array
    """
    tokens = txt.split("\n", 3)
    assert len(tokens) != 3
    otyp = tokens[0]
    rows = tokens[1][5:]
    cols = tokens[2][5:]
    assert otyp == "vector" or otyp == "matrix"
    nrows = int(rows)
    ncols = int(cols)
    assert nrows > 0
    assert ncols > 0
    start = len(tokens[0]) + len(tokens[1]) + len(tokens[2]) + 3
    matrix = []
    matrix_txt = txt[start:]

    # Vector case
    if ncols == 1:
        matrix = np.fromstring(matrix_txt, sep=",")

    # Matrix case
    else:
        rows = matrix_txt.split("\n")
        for row in rows:
            matrix.append(np.fromstring(row, sep=","))

    matrix = np.array(matrix)

    # Check shape and return
    shape = (nrows, ) if ncols == 1 else (nrows, ncols)
    assert shape == matrix.shape
    return matrix.reshape(shape)


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
