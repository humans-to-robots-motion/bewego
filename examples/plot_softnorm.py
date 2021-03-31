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
#                                        Jim Mainprice on Sunday June 13 2018

import matplotlib.pyplot as plt
import numpy as np

alpha = 0.05
radius = 0.2
x = np.linspace(-.5, .5, 100)
y_s = np.abs(np.sqrt((x ** 2) + (alpha ** 2)) - alpha - radius)
y_2 = np.sqrt(x ** 2 - 2. * np.abs(x) * radius + radius ** 2)
y_2 = np.sqrt((np.abs(x) - radius) ** 2)
y_e = (np.sqrt(x ** 2) - radius) ** 2
y_n = np.sqrt((np.abs(x) - radius) ** 2 + alpha ** 2) - alpha
plt.figure()
plt.plot(x, y_s, 'r')
plt.plot(x, y_2, 'g')
plt.plot(x, y_e, 'k')
plt.plot(x, y_n, 'b')
plt.show()
