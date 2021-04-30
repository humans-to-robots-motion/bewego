# bewego

Library and python bindings to implement fast optimization
algorithms based on gradient descent


# install and test

In an python environment. The software works for ubuntu 18.04.

Download with pybind11

    git clone --recursive git@github.com:humans-to-robots-motion/bewego.git
    git clone git@github.com:humans-to-robots-motion/pyrieef.git
    cd pyrieef pip install -r requirements.txt
    cd ../bewego

Compile and execute tests (in bewego)

    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
             -DWITH_IPOPT=true
    make
    make test
    pytest ../tests


TODO

- Need to clearly state dependencies and
  installation proceedure (pyrieef, pybullet_robots, etc.)
- Right now they have to be cloned next to bewego
- Add dependency to ipopt 
