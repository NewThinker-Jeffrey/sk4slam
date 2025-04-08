#include "sk4slam_cpp/timer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

TEST(TestTimer, schedule) {
  using sk4slam::Timer;
  std::string str;

  Timer::getNamed("default")->scheduleOnce(
      [&str]() {
        str += ".";
        LOGI("scheduleOnce: str = %s", str.c_str());
      },
      100);

  Timer::getNamed("default")->schedule(
      [&str]() {
        str += std::to_string(str.length());
        LOGI("schedule: str = %s", str.c_str());
        return str.length() < 5;
      },
      2.5);

  ASSERT_EQ(Timer::getNamed("default")->numThreads(), 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  LOGI("final: str = %s", str.c_str());
  ASSERT_EQ(str, "01234.");
}

SK4SLAM_UNITTEST_ENTRYPOINT
