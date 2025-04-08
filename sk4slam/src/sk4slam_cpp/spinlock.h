#pragma once

#include <atomic>

namespace sk4slam {

class Spinlock {
 public:
  Spinlock() : lock_flag(ATOMIC_FLAG_INIT) {}

  bool try_lock() {
    // About 'memory_order_acquire':
    //   - No reads or writes in the current thread can be reordered before
    //     this load.
    //   - All writes in other threads that release the same atomic variable
    //     are visible in the current thread
    return !lock_flag.test_and_set(std::memory_order_acquire);
  }

  void lock(const size_t holdon_loops = 32) {
    size_t loops = holdon_loops;
    while (!try_lock()) {
      if (loops-- == 0) {
        std::this_thread::yield();
        loops = holdon_loops;
      }
    }
  }

  void unlock() {
    // About 'memory_order_release':
    //   - No reads or writes in the current thread can be reordered after
    //     this store.
    //   - All writes in the current thread are visible in other threads
    //     that acquire the same atomic variable
    lock_flag.clear(std::memory_order_release);
  }

 private:
  std::atomic_flag lock_flag;
};

class alignas(8) RWSpinlock {
 public:
  using CounterType = uint64_t;
  static constexpr CounterType kWriting = (1ull << 63);
  RWSpinlock() : lock_count_(0) {}

  bool try_lock() {
    uint64_t writable = 0;
    return lock_count_.compare_exchange_strong(
        writable, kWriting, std::memory_order_acq_rel,
        std::memory_order_relaxed);
  }

  bool try_lock_shared() {
    uint64_t old_count = lock_count_.fetch_add(1, std::memory_order_acq_rel);
    if (old_count & kWriting) {
      lock_count_.fetch_sub(1, std::memory_order_relaxed);
      return false;
    } else {
      return true;
    }
  }

  void lock(const size_t holdon_loops = 32) {
    size_t loops = holdon_loops;
    while (!try_lock()) {
      if (loops-- == 0) {
        std::this_thread::yield();
        loops = holdon_loops;
      }
    }
  }

  void lock_shared(const size_t holdon_loops = 32) {
    size_t loops = holdon_loops;
    while (!try_lock_shared()) {
      if (loops-- == 0) {
        std::this_thread::yield();
        loops = holdon_loops;
      }
    }
  }

  void unlock() {
    lock_count_.fetch_and(~kWriting, std::memory_order_release);
  }

  void unlock_shared() {
    lock_count_.fetch_sub(1, std::memory_order_release);
  }

 private:
  alignas(8) std::atomic<CounterType> lock_count_;
};

}  // namespace sk4slam
