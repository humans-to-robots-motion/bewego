cmake_minimum_required(VERSION 2.8.3)
project(bewego)

############################
# Options                  #
############################
option(BEWEGO_BUILD_GPU "Compile CUDA enabled trackers" ON)

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
# add_definitions(${EIGEN3_DEFINITIONS})

# GPU libs
set(GLEW_DIR ${CMAKE_MODULE_PATH})
find_package(CUDA QUIET)
find_package(GLEW QUIET)
find_package(OpenGL QUIET)

if(BEWEGO_BUILD_GPU AND CUDA_FOUND AND GLEW_FOUND AND OPENGL_FOUND)
  include_directories(${GLEW_INCLUDE_DIRS})
  include_directories(${OpenGL_INCLUDE_DIRS})

  link_directories(${OpenGL_LIBRARY_DIRS})
  link_directories(${GLEW_LIBRARY_DIRS})

  add_definitions(${OpenGL_DEFINITIONS})
  add_definitions(${GLEW_DEFINITIONS})

  # enable cuda debug information with -g -G -O0, to use with cuda-dbg use
  # --ptxas-options=-v to see number of registers, local, shared and constant
  # memory used in kernels
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O2 -arch=sm_30)
  list(APPEND bewego_LIBRARIES ${bewego_LIBRARY_GPU} ${CUDA_CUDART_LIBRARY})

  # activate gpu implementations
  add_definitions(-DBEWEGO_BUILD_GPU=1)
  set(BEWEGO_GPU_SUPPORT "YES")
  message(STATUS "Found CUDA version ${CUDA_VERSION_STRING}")
else(BEWEGO_BUILD_GPU AND CUDA_FOUND AND GLEW_FOUND AND OPENGL_FOUND)
  set(BEWEGO_GPU_SUPPORT "NO")
  if(BEWEGO_BUILD_GPU)
    message(WARNING "No CUDA support. Deactivating GPU implementation")
  endif(BEWEGO_BUILD_GPU)
  set(BEWEGO_BUILD_GPU OFF)
endif(BEWEGO_BUILD_GPU AND CUDA_FOUND AND GLEW_FOUND AND OPENGL_FOUND)

############################
# Library info summary     #
############################
#include(cmake/info.cmake)

#info_yesorno(CUDA_FOUND BEWEGO_FOUND_CUDA)
#info_yesorno(GLEW_FOUND BEWEGO_FOUND_GLEW)
#info_yesorno(OPENGL_FOUND BEWEGO_FOUND_OPENGL)

#info_begin()
#  info_project("::bewego:: Differentiable Kinetics Library" ${PROJECT_VERSION})
#  info_header("Setup")
#  info_item("Found CUDA" "${BEWEGO_FOUND_CUDA}")
#  info_item("Found GLEW" "${BEWEGO_FOUND_GLEW}")
#  info_item("Found OpenGL" "${BEWEGO_FOUND_OPENGL}")
#  info_item("Compiling with GPU Support" "${BEWEGO_GPU_SUPPORT}")
#info_end()

############################
# Dependencies             #
############################
# Use catkin of available otherwise fall back to native cmake
find_package(catkin QUIET)

if(BEWEGO_BUILD_GPU)
  list(APPEND bewego_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  list(APPEND bewego_INCLUDE_DIRS ${CUDA_CUT_INCLUDE_DIRS})
  list(APPEND bewego_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS})
  list(APPEND bewego_INCLUDE_DIRS ${OpenGL_INCLUDE_DIR})
endif(BEWEGO_BUILD_GPU)

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
else(catkin_FOUND)
  # find_package(fl REQUIRED)
  # find_package(osr REQUIRED)

  list(APPEND bewego_INCLUDE_DIRS ${fl_INCLUDE_DIRS})
  list(APPEND bewego_INCLUDE_DIRS ${osr_INCLUDE_DIRS})
endif(catkin_FOUND)

############################
## bewego library            #
############################
include_directories(${bewego_INCLUDE_DIRS})

set(bewego_SOURCE_DIR source/${PROJECT_NAME})

file(GLOB_RECURSE bewego_HEADERS
    ${bewego_SOURCE_DIR}/*.hpp
    ${bewego_SOURCE_DIR}/*.h)

# Build bewego library
set(bewego_SOURCES    
    ${bewego_SOURCE_DIR}/differentiable_map.cpp
)

add_library(${bewego_LIBRARY} SHARED
    ${bewego_HEADERS}
    ${bewego_SOURCES})

target_link_libraries(${bewego_LIBRARY}
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES})

############################
# Tests                    #
############################
# enable_testing()
# include(${CMAKE_MODULE_PATH}/gtest.cmake)
# include(utests.cmake)