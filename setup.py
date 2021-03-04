#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup( 
    packages=['pybewego'],
    package_dir={'': '.'},
    package_data={'pybewego': ['_pybewego.cpython-38-x86_64-linux-gnu.so']},
    include_package_data=True
)

setup(**d)
