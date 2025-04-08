#include "sk4slam_cpp/work_queue.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

TEST(TestWorkQueue, SingleThreaded) {
  std::string str;
  using sk4slam::Mutex;
  using sk4slam::UniqueLock;
  Mutex mutex;
  sk4slam::WorkQueue<int> work_queue([&str, &mutex](int v) {
    UniqueLock lock(mutex);
    str += std::to_string(v);
  });

  work_queue.enqueue(1);
  work_queue.enqueue(2);
  work_queue.enqueue(3);

  work_queue.waitUntilAllJobsDone();
  LOGI("str=%s", str.c_str());
  ASSERT_EQ(str, "123");
}

TEST(TestWorkQueue, NoOrder) {
  std::string str;
  using sk4slam::Mutex;
  using sk4slam::UniqueLock;
  Mutex mutex;
  sk4slam::WorkQueue<int> work_queue(
      [&str, &mutex](int v) {
        UniqueLock lock(mutex);
        str += std::to_string(v);
      },
      "", 4);

  for (int i = 0; i < 10; i++) {
    work_queue.enqueue(i);
  }

  work_queue.waitUntilAllJobsDone();
  LOGI("str=%s", str.c_str());
  for (int i = 0; i < 10; i++) {
    ASSERT_TRUE(str.find(std::to_string(i)) != std::string::npos);
  }
}

TEST(TestWorkQueue, TaskQueue) {
  std::string str;
  using sk4slam::Mutex;
  using sk4slam::UniqueLock;
  Mutex mutex;

  sk4slam::TaskQueue task_queue("", 4);
  for (int i = 0; i < 10; i++) {
    task_queue.enqueue([&str, &mutex, i]() {
      UniqueLock lock(mutex);
      str += std::to_string(i);
    });
  }

  task_queue.waitUntilAllJobsDone();
  LOGI("str=%s", str.c_str());
  for (int i = 0; i < 10; i++) {
    ASSERT_TRUE(str.find(std::to_string(i)) != std::string::npos);
  }
}

TEST(TestWorkQueue, FifoOrder) {
  std::string ready_order;
  std::string output_order;
  using sk4slam::Mutex;
  using sk4slam::UniqueLock;
  Mutex mutex;

  sk4slam::Logging::setVerbose("ALL");
  sk4slam::WorkQueueFIFO<int, int> work_queue(
      [&ready_order, &mutex](int v) -> int {
        std::this_thread::sleep_for(std::chrono::milliseconds(v * 100));
        UniqueLock lock(mutex);
        ready_order += std::to_string(v);
        return v;
      },
      [&output_order, &mutex](int output) {
        UniqueLock lock(mutex);
        output_order += std::to_string(output);
      },
      "", 4);

  LOGI("enqueuing tasks ...");
  work_queue.enqueue(4);
  work_queue.enqueue(3);
  work_queue.enqueue(2);
  work_queue.enqueue(1);
  LOGI("waiting tasks ...");

  work_queue.waitUntilAllJobsDone();

  LOGI("ready_order=%s", ready_order.c_str());
  LOGI("output_order=%s", output_order.c_str());
  ASSERT_EQ(ready_order, "1234");
  ASSERT_EQ(output_order, "4321");
}

TEST(TestWorkQueue, FifoOrder2) {
  std::string output_order;
  using sk4slam::Mutex;
  using sk4slam::UniqueLock;
  Mutex mutex;

  sk4slam::Logging::setVerbose("ALL");
  sk4slam::WorkQueueFIFO<int, char> work_queue(
      [](int v) -> char { return 'a' + v; },
      [&output_order, &mutex](char output) {
        UniqueLock lock(mutex);
        output_order += output;
      },
      "", 4);

  LOGI("enqueuing tasks ...");
  work_queue.enqueue(10);
  work_queue.enqueue(9);
  work_queue.enqueue(8);
  work_queue.enqueue(7);
  work_queue.enqueue(6);
  work_queue.enqueue(5);
  work_queue.enqueue(4);
  work_queue.enqueue(3);
  work_queue.enqueue(2);
  work_queue.enqueue(1);
  work_queue.enqueue(0);
  LOGI("waiting tasks ...");

  work_queue.waitUntilAllJobsDone();
  LOGI("output_order=%s", output_order.c_str());
  ASSERT_EQ(output_order, "kjihgfedcba");
}

SK4SLAM_UNITTEST_ENTRYPOINT
