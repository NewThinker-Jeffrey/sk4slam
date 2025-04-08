#include "sk4slam_basic/logging.h"

#include <cmath>

#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"

TEST(TestLogging, SimpleLog) {
  LOGI(BLUE "hello world!");
  LOGI("hello world! %d", 2);

  using sk4slam::Oss;
  Oss oss;
  oss << M_PI;
  LOGW("hello world! pi=%s", oss.str().c_str());
}

TEST(TestLogging, Assert) {
  int a = 1;
  int b = 1;
  LOGE_ASSERT(a == b, "a=%d, b=%d", a, b);
  ASSERT(a == b);

  try {
    b = 2;
    LOGE_ASSERT(a == b, "a=%d, b=%d", a, b);
  } catch (const std::runtime_error& e) {
    LOGI("runtime_error catched: %s", e.what());
  }

  try {
    b = 2;
    ASSERT(a == b);
  } catch (const std::exception& e) {
    LOGI("exception catched: %s", e.what());
  }
}

TEST(TestLogging, Exception) {
  constexpr int except_value = 555;
  auto return_a_value = [](int v) {
    if (v == except_value) {
      throw std::runtime_error("get except_value!");
    }
    LOGI("return_a_value %d", v);
    return v;
  };

  auto func_void = []() { LOGI("running func_void()"); };

  TRACE_EXCEPTION(func_void());

  int a = TRACE_EXCEPTION(return_a_value(1));

  bool rethrowed;

  rethrowed = false;
  try {
    int a = TRACE_EXCEPTION(return_a_value(except_value));
  } catch (const std::runtime_error& e) {
    LOGI("Exception rethrowed as expected");
    rethrowed = true;
  }
  ASSERT_TRUE(rethrowed);

  rethrowed = false;
  try {
    int a = LOGE_EXCEPTION(
        return_a_value(except_value), "except_value = %d", except_value);
  } catch (const std::runtime_error& e) {
    LOGI("Exception rethrowed as expected");
    rethrowed = true;
  }
  ASSERT_TRUE(rethrowed);
}

SK4SLAM_UNITTEST_ENTRYPOINT
