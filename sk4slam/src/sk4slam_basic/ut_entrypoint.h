#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"

#define SK4SLAM_UNITTEST_ENTRYPOINT                         \
  int main(int argc, char** argv) {                         \
    ::testing::InitGoogleTest(&argc, argv);                 \
    google::InitGoogleLogging(argv[0]);                     \
    google::ParseCommandLineFlags(&argc, &argv, false);     \
    google::InstallFailureSignalHandler();                  \
    ::testing::FLAGS_gtest_death_test_style = "threadsafe"; \
    FLAGS_alsologtostderr = true;                           \
    FLAGS_colorlogtostderr = true;                          \
    sk4slam::Logging::setVerbose("ALL");                    \
    return RUN_ALL_TESTS();                                 \
  }

// Make the eclipse parser silent.
#ifndef TYPED_TEST
#define TYPED_TEST(x, y) int x##y()
#endif
#ifndef TYPED_TEST_CASE
#define TYPED_TEST_CASE(x, y) int x##y()
#endif
