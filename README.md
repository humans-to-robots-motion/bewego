# bewego

Library and python bindings to implement fast optimization
algorithms based on gradient descent


Download with pybind11

    git clone --recursive git@github.com:humans-to-robots-motion/bewego.git


Compile and execute tests

    
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYBIND11_PYTHON_VERSION=3.5
    make
    make test
    pytest ../tests


TODO

- Need to clearly state dependencies and
  installation proceedure (pyrieef, pybullet_robots, etc.)
- Right now they have to be cloned next to bewego
