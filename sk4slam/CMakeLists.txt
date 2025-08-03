
# Check if SK4SLAM_ROS_VERSION is set and choose build tools accordingly
if (NOT DEFINED ENV{SK4SLAM_ROS_VERSION})
  message(STATUS "No SK4SLAM_ROS_VERSION environment variable set, building with cmake only.")
  set(SK4SLAM_ROS_VERSION "none")  # Indicating no ROS
else()
  set(SK4SLAM_ROS_VERSION $ENV{SK4SLAM_ROS_VERSION})
  message(STATUS "Using ROS version: ${SK4SLAM_ROS_VERSION}")
endif()

if (SK4SLAM_ROS_VERSION STREQUAL "2")
  cmake_minimum_required(VERSION 3.22)
else()
  cmake_minimum_required(VERSION 3.5)
endif()

project(sk4slam)

# Define the options for building Python interface and unit tests
option(BUILD_SK4SLAM_PY "Whether to build python interface" ON)
option(BUILD_SK4SLAM_TESTS "Whether to build tests" ON)

# Check if SK4SLAM_ROS_VERSION is set and choose build tools accordingly
if (NOT DEFINED ENV{SK4SLAM_ROS_VERSION})
  message(STATUS "No SK4SLAM_ROS_VERSION environment variable set, building with cmake only.")
  set(SK4SLAM_ROS_VERSION "none")  # Indicating no ROS
else()
  set(SK4SLAM_ROS_VERSION $ENV{SK4SLAM_ROS_VERSION})
  message(STATUS "Using ROS version: ${SK4SLAM_ROS_VERSION}")
endif()

# Add catkin or ament support for ROS1/ROS2 build
if (SK4SLAM_ROS_VERSION STREQUAL "1")
  find_package(catkin REQUIRED
    # COMPONENTS  # We don't need any components for this package.
    # roscpp std_msgs
  )
elseif (SK4SLAM_ROS_VERSION STREQUAL "2")
  find_package(ament_cmake REQUIRED)
  # find_package(rclcpp REQUIRED)  # ROS2's rclcpp for C++ nodes
  # find_package(std_msgs REQUIRED)  # ROS2's standard message types
else()
  message(STATUS "No ROS environment detected, building with cmake only.")
endif()

# Define SUBDIRLIST macro for listing subdirectories
MACRO(SUBDIRLIST result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
ENDMACRO()

# Project and build setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SK4SLAM_SRC ${PROJECT_SOURCE_DIR})
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "SK4SLAM_SRC: ${SK4SLAM_SRC}")

set(SK4SLAM_INCLUDE_DIR ${SK4SLAM_SRC}/src)

#### Find dependencies
find_package(Eigen3 REQUIRED)
list(APPEND dependencies Eigen3)
message(STATUS "EIGEN3_FOUND: ${EIGEN3_FOUND}")
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

find_package(OpenCV 3 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 REQUIRED)
    # list(APPEND dependencies OpenCV4)
    list(APPEND dependencies OpenCV)
    message(STATUS "Use OpenCV 4")
else()
    # list(APPEND dependencies OpenCV3)
    list(APPEND dependencies OpenCV)
    message(STATUS "Use OpenCV 3")
endif ()
message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")

find_package(yaml-cpp REQUIRED)
list(APPEND dependencies yaml-cpp)
message(STATUS "yaml-cpp_FOUND: ${yaml-cpp_FOUND}")
message(STATUS "YAML_INCLUDE_DIRS: ${YAML_INCLUDE_DIRS}")  # empty
message(STATUS "YAML_CPP_LIBRARIES: ${YAML_CPP_LIBRARIES}")

find_package(Ceres REQUIRED)
list(APPEND dependencies Ceres)
message(STATUS "CERES_FOUND: ${CERES_FOUND}")
message(STATUS "CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS}")
message(STATUS "CERES_LIBRARIES: ${CERES_LIBRARIES}")

find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
list(APPEND dependencies GTSAM GTSAM_UNSTABLE)
set(gtsam_LIB gtsam)
set(gtsam_unstable_LIB gtsam_unstable)
message(STATUS "GTSAM_FOUND: ${GTSAM_FOUND}")
message(STATUS "GTSAM_INCLUDE_DIR: ${GTSAM_INCLUDE_DIR}")
message(STATUS "gtsam_LIB: ${gtsam_LIB}")
message(STATUS "gtsam_unstable_LIB: ${gtsam_unstable_LIB}")

# Only find testing dependencies if BUILD_SK4SLAM_TESTS is enabled
if (BUILD_SK4SLAM_TESTS)
  find_package(gflags REQUIRED)
  find_package(GTest REQUIRED)

  # Directly call `find_package(glog REQUIRED)` will fail!
  # We have to use pkg_config to find glog for now. See:
  # https://github.com/google/glog/issues/519#issuecomment-1014378289
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(glog REQUIRED libglog)

  message(STATUS "glog_FOUND: ${glog_FOUND}")
  message(STATUS "glog_INCLUDE_DIRS: ${glog_INCLUDE_DIRS}")
  message(STATUS "glog_LIBRARIES: ${glog_LIBRARIES}")
  message(STATUS "gflags_FOUND: ${gflags_FOUND}")
  message(STATUS "gflags_INCLUDE_DIR: ${gflags_INCLUDE_DIR}")
  message(STATUS "gflags_LIBRARIES: ${gflags_LIBRARIES}")
  message(STATUS "GTest_FOUND: ${GTest_FOUND}")
  message(STATUS "GTEST_INCLUDE_DIRS: ${GTEST_INCLUDE_DIRS}")
  message(STATUS "GTEST_LIBRARIES: ${GTEST_LIBRARIES}")

  if (SK4SLAM_ROS_VERSION STREQUAL "2")
    find_package(ament_cmake_gtest REQUIRED)  # needed by ament_add_gtest()?
  endif()
endif()

#### Collect dependencies for include directories and libraries
list(APPEND dependency_include_dirs
        ${glog_INCLUDE_DIRS}
        ${gflags_INCLUDE_DIR}
        ${GTEST_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${YAML_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${GTSAM_INCLUDE_DIR}
)

list(APPEND dependency_libraries
        ${glog_LIBRARIES}
        ${gflags_LIBRARIES}
        ${GTEST_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${CERES_LIBRARIES}
        ${gtsam_LIB}
        ${gtsam_unstable_LIB}
        atomic
)

message(STATUS "dependencies: ${dependencies}")
message(STATUS "dependency_include_dirs: ${dependency_include_dirs}")
message(STATUS "dependency_libraries:    ${dependency_libraries}")

if (SK4SLAM_ROS_VERSION STREQUAL "1")
  catkin_package(
    # CATKIN_DEPENDS  # we don't have any catkin depend for this package
    #   roscpp rosbag tf std_msgs geometry_msgs
    #   sensor_msgs nav_msgs visualization_msgs
    #   image_transport cv_bridge
    DEPENDS ${dependencies}
    INCLUDE_DIRS ${SK4SLAM_INCLUDE_DIR}
    LIBRARIES ${PROJECT_NAME}
    # CFG_EXTRAS  # for future
  )
elseif (SK4SLAM_ROS_VERSION STREQUAL "2")
  ament_export_include_directories(${SK4SLAM_INCLUDE_DIR})
  ament_export_libraries(${PROJECT_NAME})
  ament_package()
else()  
  message(STATUS "")
endif()

#### Build sk4slam library
file(GLOB_RECURSE LIBRARY_HEADERS "${SK4SLAM_SRC}/src/*.h")
file(GLOB_RECURSE LIBRARY_SOURCES "${SK4SLAM_SRC}/src/*.cc")
list(FILTER LIBRARY_SOURCES EXCLUDE REGEX ".*_test\\.cc$")
message(STATUS "LIBRARY_HEADERS: ${LIBRARY_HEADERS}")
message(STATUS "LIBRARY_SOURCES: ${LIBRARY_SOURCES}")

include_directories(${SK4SLAM_INCLUDE_DIR} ${dependency_include_dirs})
add_library(${PROJECT_NAME} SHARED ${LIBRARY_HEADERS} ${LIBRARY_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${dependency_libraries})

# Install lib
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

# Install headers
install(DIRECTORY "${SK4SLAM_SRC}/src" # source directory
        DESTINATION "include" # target directory
        FILES_MATCHING
        # PATTERN "*/3rd_party/*" EXCLUDE  # exclude all 3rd_party code
        # PATTERN "*/third_party/*" EXCLUDE  # exclude all third_party code
        PATTERN "*.h"  # select header files
)

#### Python interfaces
if (BUILD_SK4SLAM_PY)
  if (SK4SLAM_ROS_VERSION STREQUAL "1")
    find_package(pybind11_catkin REQUIRED)
    include_directories(${pybind11_catkin_INCLUDE_DIRS})
  elseif (SK4SLAM_ROS_VERSION STREQUAL "2")
    find_package(pybind11 REQUIRED)
    include_directories(${pybind11_INCLUDE_DIRS})
  else()
    find_package(pybind11 REQUIRED)
    include_directories(${pybind11_INCLUDE_DIRS})
  endif()

  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

  SUBDIRLIST(PYBINDS ${SK4SLAM_SRC}/python_interface)
  foreach(py_interface ${PYBINDS})
      file(GLOB_RECURSE py_interface_headers "${SK4SLAM_SRC}/python_interface/${py_interface}/*.h")
      file(GLOB_RECURSE py_interface_sources "${SK4SLAM_SRC}/python_interface/${py_interface}/*.cc")
      pybind11_add_module(${py_interface} ${py_interface_headers} ${py_interface_sources})
      target_link_libraries(${py_interface} PRIVATE pybind11::module ${PROJECT_NAME})
  endforeach()
  install(TARGETS ${PROJECT_NAME} DESTINATION lib)
endif()

# Build tests
if (BUILD_SK4SLAM_TESTS)
  # Ensure TEST_SOURCES is populated with all _test.cc files
  file(GLOB_RECURSE TEST_SOURCES "${SK4SLAM_SRC}/src/*_test.cc")

  foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WLE)
    message("build ${TEST_NAME} ...")

    if (SK4SLAM_ROS_VERSION STREQUAL "1")
        catkin_add_gtest(${TEST_NAME} ${TEST_SOURCE})
    elseif (SK4SLAM_ROS_VERSION STREQUAL "2")
        ament_add_gtest(${TEST_NAME} ${TEST_SOURCE})
    else()
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        add_dependencies(${TEST_NAME} ${PROJECT_NAME})
    endif()

    target_link_libraries(${TEST_NAME} ${PROJECT_NAME})
    install(TARGETS ${TEST_NAME} DESTINATION tests)
  endforeach()
endif()
