#include "sk4slam_cpp/circular_queue.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_cpp/thread_pool.h"

TEST(TestCircularQueue, SingleThreaded) {
  using sk4slam::CircularQueue;
  CircularQueue<int> queue(1024);
  for (int i = 0; i < 1024; ++i) {
    bool push_ok = queue.push(i);
    if (!push_ok) {
      LOGE("Failed to push %d to queue.", i);
      FAIL();
    }
  }
  ASSERT_FALSE(queue.push(1024));

  int pop;
  for (int i = 0; i < 1024; ++i) {
    bool pop_ok = queue.pop(pop);
    if (!pop_ok) {
      LOGE("Failed to pop %d from queue.", i);
      FAIL();
    }
    ASSERT_EQ(pop, i);
  }
  ASSERT_FALSE(queue.pop(pop));

  ASSERT_TRUE(queue.push(1024));
  ASSERT_TRUE(queue.pop(pop));
  ASSERT_EQ(pop, 1024);
}

TEST(TestCircularQueue, MultiThreaded) {
  using sk4slam::CircularQueue;
  using sk4slam::Mutex;
  using sk4slam::ThreadPool;
  using sk4slam::UniqueLock;

  ThreadPool::createNamed("push", 4);
  ThreadPool::createNamed("pop", 4);
  CircularQueue<int> queue(1024);

  Mutex mtx;
  std::set<int> pushed_nums;
  std::set<int> popped_nums;
  const int max_wait_loops = 10000000;

  auto push = [&](int start, int n) {
    for (int i = start; i < start + n; ++i) {
      int loop = 0;
      bool push_ok = false;
      // while ((++loop) < max_wait_loops) {
      while (1) {
        push_ok = queue.push(i);
        if (push_ok) {
          break;
        }
      }
      if (!push_ok) {
        LOGE("Failed to push %d to queue.", i);
        FAIL();
      } else {
        if (loop > 1) {
          LOGI("Tried %d times to push %d to queue.", loop, i);
        }
        UniqueLock lock(mtx);
        EXPECT_TRUE(pushed_nums.insert(i).second);
      }
    }
  };

  auto pop = [&](int n) {
    int pop;
    for (int i = 0; i < n; ++i) {
      int loop = 0;
      bool pop_ok = false;
      // while ((++loop) < max_wait_loops) {
      while (1) {
        pop_ok = queue.pop(pop);
        if (pop_ok) {
          break;
        }
      }
      if (!pop_ok) {
        LOGE("Failed to pop from queue. i=%d", i);
        FAIL();
      } else {
        UniqueLock lock(mtx);
        EXPECT_TRUE(popped_nums.insert(pop).second);
      }
    }
  };

  ThreadPool::getNamed("push")->schedule([&]() { push(0, 1000); });
  ThreadPool::getNamed("push")->schedule([&]() { push(1000, 1000); });
  ThreadPool::getNamed("push")->schedule([&]() { push(2000, 1000); });
  ThreadPool::getNamed("push")->schedule([&]() { push(3000, 1000); });
  ThreadPool::getNamed("pop")->schedule([&]() { pop(1000); });
  ThreadPool::getNamed("pop")->schedule([&]() { pop(1000); });
  ThreadPool::getNamed("pop")->schedule([&]() { pop(1000); });
  ThreadPool::getNamed("pop")->schedule([&]() { pop(1000); });

  ThreadPool::getNamed("push")->waitUntilAllTasksDone();
  ThreadPool::getNamed("pop")->waitUntilAllTasksDone();
  LOGI("Total pushed %d", pushed_nums.size());
  LOGI("Total popped %d", popped_nums.size());
  LOGI("queue remain size %d", queue.size());

  ASSERT_EQ(pushed_nums.size(), popped_nums.size());
  if (pushed_nums != popped_nums) {
    LOGE("pushed_nums != popped_nums");
    FAIL();
  }
}

SK4SLAM_UNITTEST_ENTRYPOINT
