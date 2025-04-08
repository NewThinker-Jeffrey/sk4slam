#include "sk4slam_basic/platform.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

TEST(TestPlatform, Print) {
  LOGI("Compiler: %s", COMPILER);
  LOGI("CPU Architecture: %s", CPU_ARCH);
  LOGI("Operating System: %s", OS);
  LOGI("C++ Standard Version: %s", CPP_STD);
}

SK4SLAM_UNITTEST_ENTRYPOINT
