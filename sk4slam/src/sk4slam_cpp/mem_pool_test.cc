#include "sk4slam_cpp/mem_pool.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"
#include "sk4slam_cpp/thread_pool.h"

TEST(TestMemPool, Ptr) {
  sk4slam::Logging::setVerbose("ALL");
  using sk4slam::MemPool;
  struct
      // alignas(8)
      alignas(16) TmpStruct {
    int a;
    int b;
    int c;
  };
  std::unique_ptr<MemPool<TmpStruct>> intpoolptr(
      MemPool<TmpStruct>::create(1024));
  MemPool<TmpStruct>& intpool = *intpoolptr;

  auto ptr = intpool.alloc();
  ASSERT_TRUE(ptr);
  (*ptr).a = 10;
  ptr->b = 20;
  ptr.get()->c = 30;
  auto short_ptr = ptr.shortPtr();
  LOGI("short_ptr is %d", short_ptr);
  ASSERT_EQ(intpool.getRefs(short_ptr), 1);

  auto ptr2 = intpool.lock(short_ptr);
  ASSERT_EQ(ptr, ptr2);
  ASSERT_EQ(ptr.get(), ptr2.get());
  ASSERT_EQ(intpool.getRefs(short_ptr), 2);

  ASSERT_EQ(ptr2->a, 10);
  ASSERT_EQ(ptr2->b, 20);
  ASSERT_EQ(ptr2->c, 30);

  ptr.reset();
  ASSERT_EQ(intpool.getRefs(short_ptr), 1);

  ptr2.reset();
  ASSERT_EQ(intpool.getRefs(short_ptr), 0);

  // short_ptr is released now, so we can't lock it anymore.
  ASSERT_FALSE(intpool.lock(short_ptr));
}

TEST(TestMemPool, SingleThreaded) {
  sk4slam::Logging::setVerbose("ALL");
  using sk4slam::MemPool;
  struct TmpStruct {
    int a;
    int b;
    int c;
  };
  using TestMemPool = MemPool<TmpStruct, 8, 24, 16, true>;
  std::unique_ptr<TestMemPool> intpoolptr(TestMemPool::create(1024));
  TestMemPool& intpool = *intpoolptr;

  using Ptr = TestMemPool::Ptr;
  using SizeType = TestMemPool::SizeType;
  std::vector<Ptr> ptrs(1024);

  for (size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(intpool.allocated(), i);
    // LOGA("allocated: %d", i);
    ptrs[i] = intpool.alloc();
    ASSERT_TRUE(ptrs[i]);
  }
  LOGA("numPages: %d", intpool.numPages());
  ASSERT_EQ(intpool.allocated(), 1024);

  for (size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(ptrs[i].refCount(), 1);
  }
  auto ptrs_cache = intpool.createPtrsCache();
  for (size_t i = 0; i < 1024; ++i) {
    // LOGI("i=%d", i);
    ASSERT_EQ(ptrs[i].refCount(), 2);
  }
  ptrs_cache.reset();
  for (size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(ptrs[i].refCount(), 1);
  }

  ASSERT_FALSE(intpool.alloc());
  ASSERT_FALSE(intpool.alloc());
  ASSERT_FALSE(intpool.alloc());

  std::vector<Ptr> tmp_ptrs;
  SizeType tmp_short_ptr;
  size_t some_indices[] = {10, 100, 300, 500, 800};
  for (auto idx : some_indices) {
    LOGI("idx = %d", idx);
    ASSERT_FALSE(intpool.alloc());  // all entries occupied.
    tmp_short_ptr = ptrs[idx].shortPtr();
    ASSERT_EQ(intpool.getRefs(tmp_short_ptr), 1);
    ptrs[idx].reset();  // release one
    ASSERT_EQ(intpool.getRefs(tmp_short_ptr), 0);
    ASSERT_EQ(intpool.allocated(), 1024 - 1);
    Ptr tmp_ptr = intpool.alloc();  // realloc
    ASSERT_TRUE(tmp_ptr);           // should be ok
    ASSERT_EQ(tmp_ptr.shortPtr(), tmp_short_ptr);
    ASSERT_EQ(intpool.getRefs(tmp_short_ptr), 1);
    tmp_ptrs.push_back(tmp_ptr);
    ASSERT_EQ(intpool.getRefs(tmp_short_ptr), 2);
  }

  for (auto& tmp_ptr : tmp_ptrs) {
    ASSERT_EQ(intpool.getRefs(tmp_ptr.shortPtr()), 1);
  }
}

SK4SLAM_UNITTEST_ENTRYPOINT
