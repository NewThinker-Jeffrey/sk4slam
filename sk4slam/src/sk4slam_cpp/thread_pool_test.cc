#include "sk4slam_cpp/thread_pool.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

TEST(TestThreadPool, Dependencies) {
  using sk4slam::Mutex;
  using sk4slam::ThreadPool;
  using sk4slam::UniqueLock;

  // Ensure there're no "pool_1" and "pool_2" before we create them.
  ThreadPool::removeNamed("pool_1");
  ThreadPool::removeNamed("pool_2");

  auto pool_1 = ThreadPool::createNamed("pool_1", 2);
  auto pool_2 = ThreadPool::createNamed("pool_2", 2);
  ASSERT_EQ(pool_1->numThreads(), 2);
  ASSERT_EQ(pool_2->numThreads(), 2);

  Mutex mtx;
  std::string str;
  auto task1 = pool_1->schedule([&str, &mtx]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    UniqueLock lock(mtx);
    str += "1";
    LOGI("task1: str = %s", str.c_str());
  });
  auto task2 = pool_2->schedule(
      [&str, &mtx]() {
        // sleep shorter than in task1.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        UniqueLock lock(mtx);
        str += "2";
        LOGI("task2: str = %s", str.c_str());
      },
      {task1});  // Let task2 depend on taks1.
  auto task3 = pool_1->schedule(
      [&str, &mtx]() {
        UniqueLock lock(mtx);
        str += "3";
        LOGI("task3: str = %s", str.c_str());
      },
      {task2});
  auto task4 = pool_2->schedule([&str, &mtx]() {
    UniqueLock lock(mtx);
    str += "4";
    LOGI("task4: str = %s", str.c_str());
  });

  pool_1->waitUntilAllTasksDone();
  pool_2->waitUntilAllTasksDone();
  LOGI("final: str = %s", str.c_str());

  ASSERT_EQ(str, "4123");
}

TEST(TestThreadPool, GetNamed) {
  using sk4slam::Mutex;
  using sk4slam::ThreadPool;
  using sk4slam::UniqueLock;

  // Ensure there're no "pool_1" and "pool_2" before we create them.
  ThreadPool::removeNamed("pool_1");
  ThreadPool::removeNamed("pool_2");

  ASSERT_EQ(ThreadPool::getNamed("pool_1")->numThreads(), 1);

  ThreadPool::createNamed("pool_2", 2);
  ASSERT_EQ(ThreadPool::getNamed("pool_2")->numThreads(), 2);

  Mutex mtx;
  std::string str;
  ThreadPool::getNamed("pool_1")->schedule([&str, &mtx]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    UniqueLock lock(mtx);
    str += "1";
    LOGI("task1: str = %s", str.c_str());
  });
  ThreadPool::getNamed("pool_2")->schedule([&str, &mtx]() {
    UniqueLock lock(mtx);
    str += "2";
    LOGI("task2: str = %s", str.c_str());
  });

  ThreadPool::getNamed("pool_1")->waitUntilAllTasksDone();
  ThreadPool::getNamed("pool_2")->waitUntilAllTasksDone();
  LOGI("final: str = %s", str.c_str());

  ASSERT_EQ(str, "21");
}

TEST(TestThreadPool, WaitTask) {
  using sk4slam::INVALID_TASK;
  using sk4slam::Mutex;
  using sk4slam::TaskID;
  using sk4slam::ThreadPool;
  using sk4slam::UniqueLock;

  // Ensure there're no thread_pool named "pool" before we create it.
  ThreadPool::removeNamed("pool");
  auto pool = ThreadPool::createNamed("pool", 2);
  ASSERT_EQ(pool->numThreads(), 2);

  Mutex mtx;
  std::vector<size_t> output;
  auto run_task = [&](size_t i) {
    UniqueLock lock(mtx);
    output.push_back(i);
    LOGI(BLUE "task %d done", i);
  };

  // Schedule N tasks
  size_t N = 20;
  std::vector<TaskID> ids(N + 1, INVALID_TASK);
  auto task_group = ThreadPool::createTaskGroupForNamed("pool");
  for (size_t i = 0; i < N; i += 2) {
    ids[i] = task_group.schedule([i, &run_task]() { run_task(i); });
    LOGI("task %d scheduled", i);
  }
  for (size_t i = 1; i < ids.size(); i += 2) {
    ids[i] = task_group.schedule(
        [i, &run_task]() { run_task(i); }, {ids[i - 1], ids[i + 1]});
    LOGI("task %d scheduled", i);
  }

  ASSERT_EQ(ids.size(), N + 1);
  ASSERT_EQ(ids[N], INVALID_TASK);
  ASSERT_FALSE(pool->wait(ids[N]));  // test wait INVALID_TASK

  pool->wait(ids[0]);  // wait one task
  task_group.wait();   //  wait all tasks
  // pool->waitTasks(ids.rbegin(), ids.rend());  // wait all tasks

  ASSERT_EQ(output.size(), N);
  for (size_t i = 1; i < N; i += 2) {
    auto it = std::find(output.begin(), output.end(), i);
    ASSERT_NE(it, output.end());
    size_t pos_i = it - output.begin();
    size_t pos_im1 =
        std::find(output.begin(), output.end(), i - 1) - output.begin();
    size_t pos_ip1 =
        std::find(output.begin(), output.end(), i + 1) - output.begin();
    LOGI("i=%d, pos_i=%d, pos_im1=%d, pos_ip1=%d", i, pos_i, pos_im1, pos_ip1);
    ASSERT_GT(pos_i, pos_im1);
    if (i + 1 < N) {
      ASSERT_GT(pos_i, pos_ip1);
    }
  }
}

SK4SLAM_UNITTEST_ENTRYPOINT
