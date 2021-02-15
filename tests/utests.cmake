find_package(gtest QUIET)
enable_testing()
if(GTEST_LIBRARY)
    message(${GTEST_INCLUDE_DIRS})
    include_directories(${GTEST_INCLUDE_DIRS})
else()
    message("gtest library not found !!! setting it up at link target")
    set(GTEST_LIBRARY gtest)
endif()

# add_executable(atomic_operators_utest tests/atomic_operators_utest.cpp)
# target_link_libraries(atomic_operators_utest ${GTEST_LIBRARY} ${PROJECT_NAME})
# include(GoogleTest)
# gtest_add_tests(TARGET atomic_operators_utest)

# add_executable(geometry_utest tests/geometry_utest.cpp)
# target_link_libraries(geometry_utest ${GTEST_LIBRARY} ${PROJECT_NAME})
# include(GoogleTest)
# gtest_add_tests(TARGET geometry_utest)

# add_executable(cost_terms_utest tests/cost_terms_utest.cpp)
# target_link_libraries(cost_terms_utest ${GTEST_LIBRARY} ${PROJECT_NAME})
# include(GoogleTest)
# gtest_add_tests(TARGET cost_terms_utest)

# add_executable(workspace_utest tests/workspace_utest.cpp)
# target_link_libraries(workspace_utest ${GTEST_LIBRARY} ${PROJECT_NAME})
# include(GoogleTest)
# gtest_add_tests(TARGET workspace_utest)


set(test_SOURCES 
    workspace_utest
    trajectory_utest
    cost_terms_utest
    geometry_utest
    atomic_operators_utest
)

foreach(test_exec ${test_SOURCES})
    add_executable(${test_exec} tests/${test_exec}.cpp)
    target_link_libraries(${test_exec} ${GTEST_LIBRARY} ${PROJECT_NAME})
    include(GoogleTest)
    gtest_add_tests(TARGET ${test_exec})
endforeach()
