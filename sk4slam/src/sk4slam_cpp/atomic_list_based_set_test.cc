#include "sk4slam_cpp/atomic_list_based_set.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_cpp/thread_pool.h"

TEST(TestListBasedSet, SingleThreaded) {
  using sk4slam::ListBasedSetForUnitTest;
  ListBasedSetForUnitTest<int> list;

  // call find() on empty list (will fail)
  size_t test_N = 100;
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_FALSE(list.find(i));
  }
  ASSERT_EQ(list.size(), 0);

  // insert
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(list.insert(i).first);
  }
  ASSERT_EQ(list.size(), test_N);

  // test foreach
  size_t foreach_count = 0;
  ASSERT_TRUE(list.foreach ([&](int& i) { ++foreach_count; }));  // NOLINT
  LOGI("foreach_count: %d", foreach_count);
  ASSERT_EQ(foreach_count, test_N);

  foreach_count = 0;
  list.foreachAlongMemory([&](int& i) { ++foreach_count; });  // NOLINT
  LOGI("foreach_count (foreachAlongMemory): %d", foreach_count);
  ASSERT_EQ(foreach_count, test_N);

  foreach_count = 0;
  const auto& const_list = list;
  ASSERT_TRUE(
      const_list.foreach ([&](const int& i) { ++foreach_count; }));  // NOLINT
  LOGI("const foreach_count: %d", foreach_count);
  ASSERT_EQ(foreach_count, test_N);

  foreach_count = 0;
  const_list.foreachAlongMemory(
      [&](const int& i) { ++foreach_count; });  // NOLINT
  LOGI("const foreach_count (foreachAlongMemory): %d", foreach_count);
  ASSERT_EQ(foreach_count, test_N);

  // then call find()
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(list.find(i));
    ASSERT_EQ(list.find(i)->data, i);
  }

  // erase
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(list.erase(i));   // erase
    ASSERT_FALSE(list.find(i));   // find after erase (will fail)
    ASSERT_FALSE(list.erase(i));  // re-erase (will fail)
  }
  ASSERT_EQ(list.size(), 0);
}

TEST(TestListBasedSet, MultiThreaded) {
  using sk4slam::ListBasedSetForUnitTest;
  using sk4slam::ThreadPool;
  sk4slam::Logging::setVerbose("ALL");

  ListBasedSetForUnitTest<int> list;
  std::atomic<int> insert_success(0);
  std::atomic<int> erase_success(0);

  const int num_insert_threads = 4;
  const int num_erase_threads = 4;

  auto print_interrupt_deletes = [&]() {
    LOGI(
        "interrupt deletes: %d,  helped deletes: %d",
        list.numInterruptedDeletes().load(),
        list.numFindHelpedDeletes().load());
  };

  auto print_list = [&]() {
    sk4slam::Oss oss;
    oss << "total " << list.size() << ": {";
    size_t n_foreach = 0;
    ASSERT(list.foreach ([&](const int& i) {  // NOLINT
      oss << i << ",";
      ++n_foreach;
    }));
    oss << "}\n";

    size_t n_foreach_memory = 0;
    oss << "foreachAlongMemory: {";
    list.foreachAlongMemory([&](const int& i) {
      oss << i << ",";
      ++n_foreach_memory;
    });
    oss << "}\n";
    LOGI(
        "%s>>> n_foreach=%d, n_foreach_memory=%d, list.size()=%d",
        oss.str().c_str(), n_foreach, n_foreach_memory, list.size());
    print_interrupt_deletes();

    ASSERT_EQ(n_foreach_memory, list.size());
    ASSERT_EQ(n_foreach, n_foreach_memory);
  };

  auto random100 = [](unsigned int* seed) {
    return rand_r(seed) % 100;
  };  // NOLINT

  auto insert_100000_times = [&](int v) {
    for (size_t i = 0; i < 100000; i++) {
      if (list.insert(v).first) {
        insert_success++;
      }
    }
  };

  auto erase_100000_times = [&](int v) {
    for (size_t i = 0; i < 100000; i++) {
      if (list.erase(v)) {
        erase_success++;
      }
    }
  };

  auto random_insert_100000_times = [&]() {
    unsigned int seed;
    for (size_t i = 0; i < 100000; i++) {
      if (list.insert(random100(&seed)).first) {
        insert_success++;
      }
    }
  };

  auto random_erase_100000_times = [&]() {
    unsigned int seed;
    for (size_t i = 0; i < 100000; i++) {
      if (list.erase(random100(&seed))) {
        erase_success++;
      }
    }
  };

  // Ensure there're no "insert" and "erase" threads before we create them.
  ThreadPool::removeNamed("insert");
  ThreadPool::removeNamed("erase");
  ThreadPool::createNamed(
      "insert", num_insert_threads > 0 ? num_insert_threads : 1);
  ThreadPool::createNamed(
      "erase", num_erase_threads > 0 ? num_erase_threads : 1);

  // insert/erase on empty list
  insert_success = 0;
  erase_success = 0;
  size_t prev_list_size = list.size();
  for (size_t i = 0; i < num_insert_threads; i++) {
    ThreadPool::getNamed("insert")->schedule([&]() { insert_100000_times(1); });
  }
  for (size_t i = 0; i < num_erase_threads; i++) {
    ThreadPool::getNamed("erase")->schedule([&]() { erase_100000_times(1); });
  }
  ThreadPool::getNamed("insert")->waitUntilAllTasksDone();
  ThreadPool::getNamed("erase")->waitUntilAllTasksDone();
  LOGI(
      "[random] insert_success = %d, erase_success = %d, list.size()=%d, "
      "prev_list_size=%d",
      insert_success.load(), erase_success.load(), list.size(), prev_list_size);
  list.find(100);
  LOGI(
      "[random] insert_success = %d, erase_success = %d, list.size()=%d, "
      "prev_list_size=%d",
      insert_success.load(), erase_success.load(), list.size(), prev_list_size);
  print_list();
  ASSERT_EQ(
      insert_success.load() - erase_success.load(),
      list.size() - prev_list_size);
  ASSERT_TRUE(list.insert(1).second);

  // insert/erase at head
  insert_success = 0;
  erase_success = 0;
  prev_list_size = list.size();
  for (size_t i = 0; i < num_insert_threads; i++) {
    ThreadPool::getNamed("insert")->schedule([&]() { insert_100000_times(0); });
  }
  for (size_t i = 0; i < num_erase_threads; i++) {
    ThreadPool::getNamed("erase")->schedule([&]() { erase_100000_times(0); });
  }
  ThreadPool::getNamed("insert")->waitUntilAllTasksDone();
  ThreadPool::getNamed("erase")->waitUntilAllTasksDone();
  LOGI(
      "[random] insert_success = %d, erase_success = %d, list.size()=%d, "
      "prev_list_size=%d",
      insert_success.load(), erase_success.load(), list.size(), prev_list_size);
  print_list();
  ASSERT_EQ(
      insert_success.load() - erase_success.load(),
      list.size() - prev_list_size);
  ASSERT_TRUE(list.insert(0).second);

  // insert/erase at tail
  insert_success = 0;
  erase_success = 0;
  prev_list_size = list.size();
  for (size_t i = 0; i < num_insert_threads; i++) {
    ThreadPool::getNamed("insert")->schedule([&]() { insert_100000_times(3); });
  }
  for (size_t i = 0; i < num_erase_threads; i++) {
    ThreadPool::getNamed("erase")->schedule([&]() { erase_100000_times(3); });
  }
  ThreadPool::getNamed("insert")->waitUntilAllTasksDone();
  ThreadPool::getNamed("erase")->waitUntilAllTasksDone();
  LOGI(
      "[random] insert_success = %d, erase_success = %d, list.size()=%d, "
      "prev_list_size=%d",
      insert_success.load(), erase_success.load(), list.size(), prev_list_size);
  print_list();
  ASSERT_EQ(
      insert_success.load() - erase_success.load(),
      list.size() - prev_list_size);
  ASSERT_TRUE(list.insert(3).second);

  // insert/erase at middle
  insert_success = 0;
  erase_success = 0;
  prev_list_size = list.size();
  for (size_t i = 0; i < num_insert_threads; i++) {
    ThreadPool::getNamed("insert")->schedule([&]() { insert_100000_times(2); });
  }
  for (size_t i = 0; i < num_erase_threads; i++) {
    ThreadPool::getNamed("erase")->schedule([&]() { erase_100000_times(2); });
  }
  ThreadPool::getNamed("insert")->waitUntilAllTasksDone();
  ThreadPool::getNamed("erase")->waitUntilAllTasksDone();
  LOGI(
      "[random] insert_success = %d, erase_success = %d, list.size()=%d, "
      "prev_list_size=%d",
      insert_success.load(), erase_success.load(), list.size(), prev_list_size);
  print_list();
  ASSERT_EQ(
      insert_success.load() - erase_success.load(),
      list.size() - prev_list_size);
  ASSERT_TRUE(list.insert(2).second);

  // check data in order, it should be 0123 now
  size_t entry = 0;
  ASSERT(list.foreach ([&](size_t v) {  // NOLINT
    ASSERT_EQ(v, entry);
    ++entry;
  }));

  // clear the list and then operate randomly
  ASSERT_TRUE(list.erase(0));
  ASSERT_TRUE(list.erase(1));
  ASSERT_TRUE(list.erase(2));
  ASSERT_TRUE(list.erase(3));
  ASSERT_EQ(list.size(), 0);

  if (num_insert_threads == 0) {
    for (size_t i = 0; i < 100; i++) {
      ASSERT_TRUE(list.insert(i).first);
    }
  }

  // insert/erase randomly
  insert_success = 0;
  erase_success = 0;
  prev_list_size = list.size();
  for (size_t i = 0; i < num_insert_threads; i++) {
    ThreadPool::getNamed("insert")->schedule(
        [&]() { random_insert_100000_times(); });
  }
  for (size_t i = 0; i < num_erase_threads; i++) {
    ThreadPool::getNamed("erase")->schedule(
        [&]() { random_erase_100000_times(); });
  }
  ThreadPool::getNamed("insert")->waitUntilAllTasksDone();
  ThreadPool::getNamed("erase")->waitUntilAllTasksDone();
  LOGI(
      "[random] insert_success = %d, erase_success = %d, list.size()=%d, "
      "prev_list_size=%d",
      insert_success.load(), erase_success.load(), list.size(), prev_list_size);
  print_list();
  ASSERT_EQ(
      insert_success.load() - erase_success.load(),
      list.size() - prev_list_size);

  // check data order
  int prev_v = -1;
  ASSERT(list.foreach ([&](int v) {  // NOLINT
    ASSERT_TRUE(v > prev_v);
    prev_v = v;
  }));
}

SK4SLAM_UNITTEST_ENTRYPOINT
