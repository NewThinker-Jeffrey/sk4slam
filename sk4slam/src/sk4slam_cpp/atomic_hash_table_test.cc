
#include "sk4slam_cpp/atomic_hash_table.h"

#include <cmath>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_cpp/thread_pool.h"

template <typename Table>
void printTable(Table& table) {
  sk4slam::Oss oss;
  oss << "total " << table.size() << ": foreachAlongBuckets {";
  size_t n_foreach_bucket = 0;
  size_t prev_bucket_idx = -1;
  table.foreachAlongBuckets([&](size_t bucket_idx, const int& i) {
    if (prev_bucket_idx != bucket_idx) {
      oss << " : (" << bucket_idx << ") ";
      prev_bucket_idx = bucket_idx;
    }
    oss << i << ",";
    ++n_foreach_bucket;
  });
  oss << "}       ";

  size_t n_foreach_memory = 0;
  oss << "foreachAlongMemory: {";
  table.foreachAlongMemory([&](const int& i) {
    oss << i << ",";
    ++n_foreach_memory;
  });
  oss << "}\n";
  LOGI(
      "%s>>> n_foreach_bucket=%d, n_foreach_memory=%d, table.size()=%d",
      oss.str().c_str(), n_foreach_bucket, n_foreach_memory, table.size());

  size_t n_elements = table.countElements();
  ASSERT_EQ(n_foreach_memory, table.size());
  ASSERT_EQ(n_foreach_memory, n_foreach_bucket);
  ASSERT_EQ(n_foreach_memory, n_elements);
}

TEST(TestHashTable, SingleThreaded) {
  sk4slam::Logging::setVerbose("ALL");
  using sk4slam::HashTableForUnitTest;
  HashTableForUnitTest<int, std::hash<int>, true> table(64);

  // call find() on empty table (will fail)
  size_t test_N = 64;
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_FALSE(table.find(i));
  }
  ASSERT_EQ(table.size(), 0);

  // insert
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(table.insert(i).first);
  }
  table.waitForRehashing();
  ASSERT_EQ(table.size(), test_N);
  printTable(table);

  int foreach_count = 0;
  table.foreachAlongMemory([&](const int& data) {  // NOLINT
    foreach_count++;
    // LOGI("foreach_count: %d, data = %d", foreach_count, data);
  });
  ASSERT_EQ(foreach_count, test_N);

  // then call find()
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(table.find(i));
    ASSERT_EQ(table.find(i)->data, i);
  }

  // erase
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(table.erase(i));   // erase
    ASSERT_FALSE(table.find(i));   // find after erase (will fail)
    ASSERT_FALSE(table.erase(i));  // re-erase (will fail)
  }
  ASSERT_EQ(table.size(), 0);
}

TEST(TestHashMap, SingleThreaded) {
  sk4slam::Logging::setVerbose("ALL");
  using sk4slam::HashMapForUnitTest;
  HashMapForUnitTest<int, double> sqrt_map;

  // call find() on empty table (will fail)
  size_t test_N = 100;
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_FALSE(sqrt_map.find(i));
  }
  ASSERT_EQ(sqrt_map.size(), 0);

  // insert
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(sqrt_map.insert(i, std::sqrt(i)).first);
  }
  ASSERT_EQ(sqrt_map.size(), test_N);

  // then call find()
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(sqrt_map.find(i));
    ASSERT_EQ(sqrt_map.find(i)->data.key, i);
    ASSERT_EQ(sqrt_map.find(i)->data.value, std::sqrt(i));
  }

  // erase
  for (size_t i = 0; i < test_N; i++) {
    ASSERT_TRUE(sqrt_map.erase(i));   // erase
    ASSERT_FALSE(sqrt_map.find(i));   // find after erase (will fail)
    ASSERT_FALSE(sqrt_map.erase(i));  // re-erase (will fail)
  }
  ASSERT_EQ(sqrt_map.size(), 0);
}

TEST(TestHashTable, MultiThreaded) {
  sk4slam::Logging::setVerbose("ALL");
  using sk4slam::HashTableForUnitTest;
  using sk4slam::ThreadPool;

  struct IdentityHash {
    size_t operator()(const int& i) const {
      return i;
    }
  };

  // HashTableForUnitTest<int, IdentityHash, true> table(
  //     1024);  // initial buckets: 1024. rehash won't be triggerred.
  HashTableForUnitTest<int, IdentityHash, true> table(
      64);  // initial buckets: 64. rehash will be triggered.
  std::atomic<int> insert_success(0);
  std::atomic<int> erase_success(0);

  const int num_insert_threads = 4;
  const int num_erase_threads = 4;

  auto random200 = [](unsigned int* seed) {
    return rand_r(seed) % 200;
  };  // NOLINT

  auto random_insert_100000_times = [&]() {
    unsigned int seed;
    for (size_t i = 0; i < 100000; i++) {
      if (i & 1) {  // for odd i's, we do pressure test for rehash()
        int v = table.getCurrentRehasingIdx() + (rand_r(&seed) % 3 - 1);
        // int v = table.getCurrentRehasingIdx();
        if (table.insert(v).first) {
          insert_success++;
        }
      } else {
        if (table.insert(random200(&seed)).first) {
          insert_success++;
        }
      }
    }
  };

  auto random_erase_100000_times = [&]() {
    unsigned int seed;
    for (size_t i = 0; i < 100000; i++) {
      if (i & 1) {  // for odd i's, we do pressure test for rehash()
        int v = table.getCurrentRehasingIdx() + (rand_r(&seed) % 3 - 1);
        if (table.erase(v)) {
          erase_success++;
        }
      } else {
        if (table.erase(random200(&seed))) {
          erase_success++;
        }
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

  // insert/erase randomly
  insert_success = 0;
  erase_success = 0;
  size_t prev_table_size = table.size();
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

  table.waitForRehashing();

  LOGI(
      "[random] insert_success = %d, erase_success = %d, table.size()=%d, "
      "prev_table_size=%d",
      insert_success.load(), erase_success.load(), table.size(),
      prev_table_size);

  printTable(table);
  ASSERT_EQ(
      insert_success.load() - erase_success.load(),
      table.size() - prev_table_size);

  ASSERT_TRUE(table.verifyRehashing());
}

SK4SLAM_UNITTEST_ENTRYPOINT
