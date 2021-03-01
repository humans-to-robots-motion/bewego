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
#                                            Jim Mainprice on Mon Mar 1 2021

import cvxopt
import numpy as np
import os

def load_qp():

    directory = os.path.abspath(os.path.dirname(__file__)) + os.sep + "qp_data"

    # Objective
    H = np.loadtxt(directory + os.sep + "Hm.csv") # P
    d = np.loadtxt(directory + os.sep + "dv.csv") # q
    c = np.loadtxt(directory + os.sep + "c.csv")

    # Inequalities
    A = np.loadtxt(directory + os.sep + "Am.csv") # G
    a = np.loadtxt(directory + os.sep + "av.csv") # h
    print(A.shape)

    # Equalities
    B = np.loadtxt(directory + os.sep + "Bm.csv") # A
    b = np.loadtxt(directory + os.sep + "bv.csv") # b

    return H, d, A, a, B, b


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


P, q, G, h, A, b = load_qp()
x = cvxopt_solve_qp(P, q, G, h, A, b)
print(x)
