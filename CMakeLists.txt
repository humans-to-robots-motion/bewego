cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0057 NEW)

project(bewego)

############################
# Flags                    #
############################
# Enable c++11 GCC 4.7 or greater required
add_definitions(-std=c++14 -fno-omit-frame-pointer -fPIC)
add_definitions(-DPROFILING_ON=1) #print profiling output
add_definitions(-Wno-deprecated-register)
# add_definitions(-Wno-deprecated-declarations)
# add_definitions(-Wall)
# add_definitions(-Wno-unused-local-typedefs)
# add_definitions(-Wno-comment)

############################
# Library Version          #
############################
# include(cmake/version.cmake)

############################
# Setup                    #
############################
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
endif()

# local variables
set(bewego_LIBRARY ${PROJECT_NAME})
set(bewego_LIBRARY_GPU ${bewego_LIBRARY}_gpu)

# parent scope variables; exported at the end
set(bewego_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/source)
set(bewego_LIBRARIES ${bewego_LIBRARY})

############################
# Dependencies             #
############################

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

############################
# Dependencies             #
############################
# Use catkin of available otherwise fall back to native cmake
find_package(catkin QUIET)

if(catkin_FOUND)
  message(STATUS "Using catkin")
  catkin_package(
    INCLUDE_DIRS
      ${bewego_INCLUDE_DIRS}
    LIBRARIES
      ${bewego_LIBRARIES}
    CATKIN_DEPENDS
    DEPENDS
      )
  list(APPEND bewego_INCLUDE_DIRS ${catkin_INCLUDE_DIRS})

  ## Uncomment this if the package has a setup.py. This macro ensures
  ## modules and global scripts declared therein get installed
  ## See http://ros.org/doc/groovy/api/catkin/html/user_guide/setup_dot_py.html
  catkin_python_setup()

  ## Tests
  catkin_add_nosetests(tests)

else(catkin_FOUND)
    message("Building without catkin")
endif(catkin_FOUND)

# message("where to compile : ${CMAKE_CURRENT_BINARY_DIR}")
# message("where to compile : ${PROJECT_SOURCE_DIR}")

############################
## bewego bindings         #
############################
set(PYBIND11_PYTHON_VERSION 3.5 CACHE STRING "")
add_subdirectory(pybind11)
pybind11_add_module(_pybewego SHARED source/bewego/bindings.cpp)
target_link_libraries(_pybewego PRIVATE bewego)
set_target_properties(_pybewego
  PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pybewego"
)

############################
## bewego library          #
############################
include_directories(${bewego_INCLUDE_DIRS})
set(bewego_SOURCE_DIR source/${PROJECT_NAME})
file(GLOB_RECURSE bewego_HEADERS 
    ${bewego_SOURCE_DIR}/*.hpp ${bewego_SOURCE_DIR}/*.h)

# Build bewego library
set(bewego_SOURCES 
    ${bewego_SOURCE_DIR}/differentiable_map.cpp
    ${bewego_SOURCE_DIR}/chrono.cpp
    ${bewego_SOURCE_DIR}/astar.cpp
    ${bewego_SOURCE_DIR}/planar_grid.cpp
    ${bewego_SOURCE_DIR}/value_iteration.cpp
    ${bewego_SOURCE_DIR}/kinematics.cpp
    ${bewego_SOURCE_DIR}/trajectory.cpp
    ${bewego_SOURCE_DIR}/cost_terms.cpp
)
add_library(${bewego_LIBRARY} SHARED
    ${bewego_HEADERS}
    ${bewego_SOURCES})

############################
# Tests                    #
############################
include(${PROJECT_SOURCE_DIR}/tests/utests.cmake)