set(MODULE_NAME numerical_optimization)
message("MODULE_NAME is ${MODULE_NAME}")

set(MODULE_DIR ${CMAKE_CURRENT_LIST_DIR})
message("MODULE_DIR is ${MODULE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake/Modules")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

find_package(IPOPT 3.11.9 REQUIRED)
if(CMAKE_MACOSX_RPATH)
string(REPLACE /coin-or # Warning here this depends on the version
       "" IPOPT_INCLUDE_DIRS_STRIP
       ${IPOPT_INCLUDE_DIRS})
else()
string(REPLACE /coin # TODO
     "" IPOPT_INCLUDE_DIRS_STRIP
     ${IPOPT_INCLUDE_DIRS})
endif()

message("IOPT include_dirs: ${IPOPT_INCLUDE_DIRS}")
message("IPOPT_INCLUDE_DIRS_STRIP: ${IPOPT_INCLUDE_DIRS_STRIP}")
message("IOPT version: ${IPOPT_VERSION}") 
message(STATUS "IPOPT_CFLAGS:  ${IPOPT_CFLAGS}")
message(STATUS "IPOPT_CFLAGS_OTHER: ${IPOPT_CFLAGS_OTHER}")

# Local directory
file(GLOB_RECURSE INC "${MODULE_DIR}/*.h")

# library
add_library(${MODULE_NAME} SHARED
    ${MODULE_DIR}/constrained_optimization_problem.cpp
    ${MODULE_DIR}/optimizer.cpp
    ${MODULE_DIR}/ipopt_problem.cpp
    ${MODULE_DIR}/ipopt_optimizer.cpp
    ${MODULE_DIR}/optimization_test_functions.cpp
    ${MODULE_DIR}/trajectory_optimization.cpp
    ${MODULE_DIR}/bewopt_planar.cpp
    ${MODULE_DIR}/bewopt_freeflyer.cpp
    ${INC})
target_include_directories(${MODULE_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/bewego>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${IPOPT_INCLUDE_DIRS_STRIP}
)
target_link_libraries(${MODULE_NAME}
  PUBLIC
    ${PROJECT_NAME}
    ${GTEST_LIBRARY}
  PRIVATE
    ${IPOPT_LIBRARIES}
    ${IPOPT_LINK_FLAGS}
)
target_compile_definitions(${MODULE_NAME}
  PRIVATE
    ${IPOPT_DEFINITIONS}
)

#################
## Tests       ##
#################

# enable_testing()


# file(GLOB_RECURSE test_SOURCES
#   RELATIVE "${MODULE_DIR}/tests/cpp-unittests" "*utest.cpp")

set(test_SOURCES
    ${MODULE_DIR}/tests/test_ipopt.cpp
    ${MODULE_DIR}/tests/test_planar_motion_optimization.cpp
    ${MODULE_DIR}/tests/test_sparse.cpp
    ${MODULE_DIR}/tests/test_trajectory_optimization.cpp
    )

foreach(test_exec ${test_SOURCES})
    message("add : ${test_exec}")
    # string(REGEX REPLACE "\\.[^.]*$" "" test_name ${test_exec})
    get_filename_component(test_name ${test_exec} NAME_WE)

    # TODO replace test names in this module
    # by _utest and have the file(GLOB_RECURSE test_SOURCES)
    # only focusing on the main test folder.

    string(REPLACE "test_" "" test_name ${test_name})
    set(test_name "${test_name}_utest")
    message("test name: ${test_name}")
    add_executable(${test_name} ${test_exec})
    target_link_libraries(
        ${test_name} 
        ${MODULE_NAME})
    target_include_directories(${test_name}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/bewego>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${IPOPT_INCLUDE_DIRS_STRIP}
        )
    target_compile_definitions(${test_name}
        PRIVATE
        ${IPOPT_DEFINITIONS}
    )
    include(GoogleTest)
    gtest_add_tests(TARGET ${test_name})
endforeach()


# ----------------------------------------------

# add_executable(sparse_utest 
#     ${MODULE_DIR}/tests/test_sparse.cpp)
# target_link_libraries(sparse_utest ${MODULE_NAME})
# include(GoogleTest)
# gtest_add_tests(TARGET sparse_utest)

# ----------------------------------------------

# add_executable(ipopt_utest
#     ${MODULE_DIR}/tests/test_ipopt.cpp)
# target_link_libraries(ipopt_utest ${MODULE_NAME})
# target_include_directories(ipopt_utest
#   PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/bewego>
#     $<INSTALL_INTERFACE:include>
#   PRIVATE
#     ${IPOPT_INCLUDE_DIRS_STRIP}
# )
# target_compile_definitions(ipopt_utest
#   PRIVATE
#     ${IPOPT_DEFINITIONS}
# )
# include(GoogleTest)
# gtest_add_tests(TARGET ipopt_utest)

# ----------------------------------------------

# add_executable(trajectory_optimization_utest
#     ${MODULE_DIR}/tests/test_trajectory_optimization.cpp)
# target_link_libraries(
#     trajectory_optimization_utest 
#     ${MODULE_NAME})
# target_include_directories(trajectory_optimization_utest
#   PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/bewego>
#     $<INSTALL_INTERFACE:include>
#   PRIVATE
#     ${IPOPT_INCLUDE_DIRS_STRIP}
#     )
# target_compile_definitions(trajectory_optimization_utest
#   PRIVATE
#     ${IPOPT_DEFINITIONS}
# )
# include(GoogleTest)
# gtest_add_tests(TARGET trajectory_optimization_utest)

#-------------------------------------------------------------------------------
# Install specifications
#-------------------------------------------------------------------------------

# install(TARGETS ${PROJECT_NAME}
#         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#         )
