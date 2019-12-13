cmake_minimum_required(VERSION 2.8.3)
project(bewego)

############################
# Flags                    #
############################
# Enable c++11 GCC 4.7 or greater required
add_definitions(-std=c++14 -fno-omit-frame-pointer -fPIC)
add_definitions(-DPROFILING_ON=1) #print profiling output

#add_definitions(-Wall)
#add_definitions(-Wno-unused-local-typedefs)
#add_definitions(-Wno-deprecated-declarations)
#add_definitions(-Wno-comment)
## for eigen-3.1.2
#add_definitions(-Wno-deprecated-register)

############################
# Library Version          #
############################
# include(cmake/version.cmake)

############################
# Setup                    #
############################
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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
set(PYBIND11_PYTHON_VERSION 2.7 CACHE STRING "")
add_subdirectory(pybind11)
pybind11_add_module(pybewego SHARED source/bewego/bindings.cpp)
target_link_libraries(pybewego PRIVATE bewego)
set_target_properties(pybewego
  PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python"
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
)
add_library(${bewego_LIBRARY} SHARED
    ${bewego_HEADERS}
    ${bewego_SOURCES})

############################
# Tests                    #
############################
# enable_testing()
# include(${CMAKE_MODULE_PATH}/gtest.cmake)
# include(utests.cmake)