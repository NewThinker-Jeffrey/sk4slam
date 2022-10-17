#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sk4slam_common/entrypoint.h>

class TestRungeKutta : public testing::Test {
 protected:
  virtual void SetUp() {
    // todo
  }

  virtual void TearDown() {
    // todo
  }
};

TEST_F(TestRungeKutta, SimpleCase) {
  // todo
  std::cout << "TestRungeKutta, SimpleCase" << std::endl;
}

SK4SLAM_UNITTEST_ENTRYPOINT
