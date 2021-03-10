set(MODULE_NAME numerical_optimization)
message("MODULE_NAME is ${MODULE_NAME}")

set(MODULE_DIR ${CMAKE_CURRENT_LIST_DIR})
message("MODULE_DIR is ${MODULE_DIR}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake/Modules")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

find_package(IPOPT 3.11.9 REQUIRED)
string(REPLACE /coin
       "" IPOPT_INCLUDE_DIRS_STRIP
       ${IPOPT_INCLUDE_DIRS})
message("IOPT include_dirs: ${IPOPT_INCLUDE_DIRS}")
message("IPOPT_INCLUDE_DIRS_STRIP: ${IPOPT_INCLUDE_DIRS_STRIP}")
message("IOPT version: ${IPOPT_VERSION}") 

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

add_executable(sparse_utest 
    ${MODULE_DIR}/tests/test_sparse.cpp)
target_link_libraries(sparse_utest ${MODULE_NAME})
include(GoogleTest)
gtest_add_tests(TARGET sparse_utest)

add_executable(ipopt_utest
    ${MODULE_DIR}/tests/test_ipopt.cpp)
target_link_libraries(ipopt_utest
    ${MODULE_NAME} 
    PRIVATE
    ${IPOPT_LIBRARIES}
    ${IPOPT_LINK_FLAGS})
target_include_directories(ipopt_utest
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/bewego>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${IPOPT_INCLUDE_DIRS_STRIP}
)
include(GoogleTest)
gtest_add_tests(TARGET ipopt_utest)

add_executable(trajectory_optimization_utest
    ${MODULE_DIR}/tests/test_trajectory_optimization.cpp)
target_link_libraries(
    trajectory_optimization_utest 
    ${MODULE_NAME}
    PRIVATE
    ${IPOPT_LIBRARIES}
    ${IPOPT_LINK_FLAGS})
target_include_directories(trajectory_optimization_utest
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/source/bewego>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${IPOPT_INCLUDE_DIRS_STRIP}
)
include(GoogleTest)
gtest_add_tests(TARGET trajectory_optimization_utest)

#-------------------------------------------------------------------------------
# Install specifications
#-------------------------------------------------------------------------------

# install(TARGETS ${PROJECT_NAME}
#         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#         )
