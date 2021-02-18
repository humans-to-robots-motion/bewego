find_package(gtest QUIET)
enable_testing()
if(GTEST_LIBRARY)
    message(${GTEST_INCLUDE_DIRS})
    include_directories(${GTEST_INCLUDE_DIRS})
else()
    message("gtest library not found !!! setting it up at link target")
    set(GTEST_LIBRARY gtest)
endif()

set(test_SOURCES 
    workspace_utest
    trajectory_utest
    cost_terms_utest
    geometry_utest
    atomic_operators_utest
    objective_utest
    interpolation_utest
)

foreach(test_exec ${test_SOURCES})
    add_executable(${test_exec} tests/${test_exec}.cpp)
    target_link_libraries(${test_exec} ${GTEST_LIBRARY} ${PROJECT_NAME})
    include(GoogleTest)
    gtest_add_tests(TARGET ${test_exec})
endforeach()
