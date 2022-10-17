cmake_minimum_required(VERSION 2.8.3)
project(sk4slam_common)

link_directories(${CATKIN_DEVEL_PREFIX}/lib)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11 -Wno-enum-compare)

cs_add_library(${PROJECT_NAME} src/dependency_tracker.cc)
target_link_libraries(${PROJECT_NAME} pthread)

# add tests
catkin_add_gtest(test_example  test/test_example.cc)
target_link_libraries(test_example ${PROJECT_NAME})

cs_install()
cs_export()