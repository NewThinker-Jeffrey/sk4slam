#include "sk4slam_basic/time.h"

#include <set>
#include <thread>
#include <unordered_set>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

using sk4slam::Duration;
using sk4slam::Time;
using sk4slam::TimeCounter;

TEST(TestTime, TimeAndDuration) {
  Time t1 = Time::now();
  std::cout << "t1 = " << t1 << std::endl;
  Time t2 = Time::now();
  std::cout << "t2 = " << t2 << std::endl;
  Duration d = t2 - t1;
  std::cout << "d = " << d << std::endl;
  std::cout << "d.nanos() = " << d.nanos() << std::endl;
  std::cout << "d.micros() = " << d.micros() << std::endl;
  std::cout << "d.millis() = " << d.millis() << std::endl;
}

TEST(TestTime, TimeCounter) {
  TimeCounter tc;
  tc.tag("checkpoint 0");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  tc.tag("checkpoint 1");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  tc.tag("checkpoint 2");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  tc.tag("checkpoint 3");

  double allow_delay_ms = 50.0;
  // double allow_delay_ms = 500.0;
  ASSERT_TRUE(tc.checkThresholds(
      {{{"checkpoint 0", "checkpoint 1"},
        Duration::Millis(100 + allow_delay_ms)},
       {{"checkpoint 1", "checkpoint 2"},
        Duration::Millis(200 + allow_delay_ms)},
       {{"checkpoint 2", "checkpoint 3"},
        Duration::Millis(300 + allow_delay_ms)}}));
  std::cout << tc.report("Test TimeCounter: ") << std::endl;
  std::cout << "Elapsed " << tc.elapsed() << std::endl;
}

SK4SLAM_UNITTEST_ENTRYPOINT
