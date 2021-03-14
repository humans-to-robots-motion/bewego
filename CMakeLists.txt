cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0057 NEW)

enable_testing()

############################
# Build Type               #
############################

if(DEFINED CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
Release RelWithDebInfo MinSizeRel.")
else()
   set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
endif()

message("Build Type: " ${CMAKE_BUILD_TYPE}) 

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
# Catkin                   #
############################
# Use catkin of available otherwise fall back to native cmake
# find_package(catkin QUIET)

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

############################
## gtest                   #
############################

# find_package(gtest QUIET)
# enable_testing()
# if(GTEST_LIBRARY)
#     message(${GTEST_INCLUDE_DIRS})
#     include_directories(${GTEST_INCLUDE_DIRS})
# else()
#     message("gtest library not found !!! setting it up at link target")
#     set(GTEST_LIBRARY gtest)
# endif()

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# Now simply link against gtest or gtest_main as needed.
set(GTEST_LIBRARY gtest_main)

# message("where to compile : ${CMAKE_CURRENT_BINARY_DIR}")
# message("where to compile : ${PROJECT_SOURCE_DIR}")

############################
## bewego library          #
############################
include_directories(${bewego_INCLUDE_DIRS})
set(bewego_SOURCE_DIR source/${PROJECT_NAME})
file(GLOB_RECURSE bewego_HEADERS 
    ${bewego_SOURCE_DIR}/*.hpp ${bewego_SOURCE_DIR}/*.h)

# Build bewego library
set(bewego_SOURCES
    ${bewego_SOURCE_DIR}/util/misc.cpp
    ${bewego_SOURCE_DIR}/util/util.cpp
    ${bewego_SOURCE_DIR}/util/chrono.cpp
    ${bewego_SOURCE_DIR}/util/interpolation.cpp
    ${bewego_SOURCE_DIR}/util/socket.cpp
    ${bewego_SOURCE_DIR}/derivatives/differentiable_map.cpp
    ${bewego_SOURCE_DIR}/derivatives/atomic_operators.cpp
    ${bewego_SOURCE_DIR}/planning/astar.cpp
    ${bewego_SOURCE_DIR}/planning/planar_grid.cpp
    ${bewego_SOURCE_DIR}/planning/value_iteration.cpp
    ${bewego_SOURCE_DIR}/motion/kinematics.cpp
    ${bewego_SOURCE_DIR}/motion/trajectory.cpp
    ${bewego_SOURCE_DIR}/motion/cost_terms.cpp
    ${bewego_SOURCE_DIR}/motion/objective.cpp
    ${bewego_SOURCE_DIR}/workspace/workspace.cpp
    ${bewego_SOURCE_DIR}/workspace/pixelmap.cpp
)
add_library(${bewego_LIBRARY} 
  SHARED
    ${bewego_HEADERS}
    ${bewego_SOURCES}
)
target_link_libraries(${bewego_LIBRARY} 
  PRIVATE 
    ${GTEST_LIBRARY}
)
# include(GoogleTest)

############################
# numerical_optimization   #
############################
if(WITH_IPOPT)
  include(${bewego_SOURCE_DIR}/numerical_optimization/numerical_optimization.cmake)
  if(NOT IPOPT_LIBRARIES)
    message(FATAL_ERROR "IPOPT library not found")
  endif()
endif()

############################
## bewego bindings         #
############################
# set(PYBIND11_PYTHON_VERSION 3.5 CACHE STRING "")
add_subdirectory(pybind11)
pybind11_add_module(_pybewego SHARED source/bewego/bindings.cpp)
if(NOT WITH_IPOPT)
    target_link_libraries(_pybewego PRIVATE bewego ${GTEST_LIBRARY})
else()
    target_include_directories(_pybewego
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/bewego>
            $<INSTALL_INTERFACE:include>
        PRIVATE
           ${IPOPT_INCLUDE_DIRS_STRIP}
    )
    target_link_libraries(_pybewego 
        PRIVATE
        bewego 
        numerical_optimization
        ${GTEST_LIBRARY}
        ${IPOPT_LIBRARIES}
        ${IPOPT_LINK_FLAGS}
    )
    message("Add IPOPT definitions to pybewego")
    target_compile_definitions(_pybewego
        PRIVATE
        ${IPOPT_DEFINITIONS}
        WITH_IPOPT)
endif()
set_target_properties(_pybewego
  PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pybewego"
)   

############################
# Tests                    #
############################
include(${PROJECT_SOURCE_DIR}/tests/utests.cmake)
