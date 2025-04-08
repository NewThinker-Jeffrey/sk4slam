#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>  // std::exit
#include <cstring>  // std::memset

#include "sk4slam_basic/logging.h"

namespace sk4slam {

template <
    typename T, bool _use_compact_valid_bits = false
    // If _use_compact_valid_bits = true, the valid_ array will
    // take less memory, i.e 1 bit per entry (vs 8 bit per entry
    // when _use_compact_valid_bits = false), but extra CAS operation
    // and memory barrier will be necessary to set/clear valid bits.
    >
class CircularQueue {
  // clang-format off
  // State transition:
  //     H = (n_pushed_ % capacity_) is the next position to push,
  //     while T = (n_popped % capacity_) is the next position to pop.
  //     (H for head, T for tail)
  // ------------------------------------------------------------
  // >>> Initial state:
  //
  //         H/T
  //          ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
  // valid_: |0|  |0|  |0|  |0|  |0|  |0|  |0|  |0|
  //
  // ------------------------------------------------------------
  // >>> Push one
  //     stage 1 (move the head pointer):
  //
  //                            old_H -> new_H
  //          ...    ↓      ↓     ↓       ↓    ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |0|     |0|  |0|  |0| ...
  //                 ↑
  //          (0/1 means the recently pushed entries might not been
  //           prepared yet which is likely to happen when calling
  //           push() in multi-threads).
  //
  //     stage 2 (copy the pushed data and mark the entry as valid):
  //
  //                              old_H    new_H
  //          ...    ↓      ↓       ↓        ↓    ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |0 -> 1|   |0|  |0|  |0| ...
  //
  // ------------------------------------------------------------
  // >>> Pop one
  //     stage 1 (move the tail pointer):
  //
  //                            old_T -> new_T
  //          ...    ↓      ↓     ↓       ↓    ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |1|     |1|  |1|  |1| ...
  //                 ↑
  //          (0/1 means the recently popped entries might not been
  //           cleared yet which is likely to happen when calling
  //           pop() in multi-threads).
  //
  //     stage 2 (clear valid):
  //
  //                              old_T    new_T
  //          ...    ↓      ↓       ↓        ↓    ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |1 -> 0|   |1|  |1|  |1| ...
  //
  // ------------------------------------------------------------
  // >>> Full state
  //     situation 1:  n_popped_ + capacity_ == n_pushed_
  //
  //                             H/T
  //          ...    ↓      ↓     ↓     ↓    ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |1|   |1|  |1|  |1| ...
  //                 ↑      ↑
  //         (recently pushed
  //          entries)
  //                              ↑     ↑    ↑    ↑
  //         (We assume that the earliest pushed entries have already
  //          been marked as valid when the head pointer goes through
  //          a loop).
  //
  //     situation 2(pseudo-full):  H is behind T && valid_[H] == 1
  //
  //                              H     ...    T
  //          ...    ↓      ↓     ↓      ↓     ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |1|   |0/1|  |1|  |1| ...
  //                 ↑      ↑
  //         (recently pushed
  //          entries)
  //                              ↑
  //         (tail pointer has moved forward, but the entry pointed by H
  //          has not been released yet (the call to pop() for that entry
  //          is in pop.stage2)
  //
  // ------------------------------------------------------------
  // >>> Empty state
  //     situation 1:  n_popped_ == n_pushed_
  //
  //                             T/H
  //          ...    ↓      ↓     ↓     ↓    ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |0|   |0|  |0|  |0| ...
  //                 ↑      ↑
  //         (recently popped
  //          entries)
  //                              ↑     ↑    ↑    ↑
  //         (These might be the unused or the earliest popped entries.
  //          We assume that the earliest popped entries have already
  //          been marked as invalid when the tail pointer goes through
  //          a loop).
  //
  //     situation 2 (pseudo-empty):  T is behind H && valid[T] == 0
  //
  //                              T    ...    H
  //          ...    ↓      ↓     ↓     ↓     ↓    ↓  ...
  // valid_:  ...  |0/1|  |0/1|  |0|  |0/1|  |0|  |0| ...
  //                 ↑      ↑
  //         (recently popped
  //          entries)
  //                              ↑
  //         (head pointer has moved forward, but the entry pointed by T
  //          has not been marked as valid yet (the call to push() for that
  //          entry is in push.stage2)
  // ------------------------------------------------------------
  // clang-format on

 public:
  explicit CircularQueue(uint64_t capacity = 1024)
      : capacity_(capacity), n_pushed_(0), n_popped_(0) {
    if ((capacity_ & (capacity_ - 1))) {
      LOGE("CircularQueue: capacity_ must be 2^N !!!");
      std::exit(EXIT_FAILURE);
    }

    mod_ = capacity_ - 1;
    data_ = new T[capacity_];
    initValidBits();
    if (n_pushed_.is_lock_free() == false ||
        n_popped_.is_lock_free() == false ||
        getAtomicValid(0).is_lock_free() == false) {
      LOGE("CircularQueue: std::atomic<uint64_t> is not lock-free !!!");
      std::exit(EXIT_FAILURE);
    }
  }

  ~CircularQueue() {
    delete[] data_;
    delete[] valid_;
  }

  bool push(T v) {
    while (true) {
      uint64_t n_pushed = n_pushed_.load(std::memory_order_acquire);
      uint64_t n_popped = n_popped_.load(std::memory_order_acquire);

      // TODO(jeffrey): Is it necesary to consider the overflow of uint64_t?
      if (n_pushed >= n_popped + capacity_) {
        ASSERT(n_pushed == n_popped + capacity_);
        return false;
      }

      uint64_t head = n_pushed & mod_;
      if (isValid(head)) {
        // If the entry is not released yet.
        return false;
      }

      // NOTE: We don't have to handle ABA problems here.
      //       Indeed in some senarios ABA problems are very likely to happen
      //       (e.g. in lock-free stacks), however it can hardly happen in our
      //       case, since looping over the range of uint64 takes an
      //       extremely long time.
      if (!n_pushed_.compare_exchange_strong(
              n_pushed, n_pushed + 1, std::memory_order_seq_cst,
              std::memory_order_relaxed)) {
        // LOGA("CircularQueue::push(): retry since CAS return false.");
        continue;
      }

      data_[head] = std::move(v);
      setValid(head);
      return true;
    }
  }

  bool pop(T& v) {
    while (true) {
      uint64_t n_popped = n_popped_.load(std::memory_order_acquire);
      uint64_t n_pushed = n_pushed_.load(std::memory_order_acquire);

      // TODO(jeffrey): Is it necesary to consider the overflow of uint64_t?
      if (n_popped >= n_pushed) {
        ASSERT(n_popped == n_pushed);
        return false;
      }

      uint64_t tail = n_popped & mod_;
      if (!isValid(tail)) {
        // If the entry is not valid yet.
        return false;
      }

      if (!n_popped_.compare_exchange_strong(
              n_popped, n_popped + 1, std::memory_order_seq_cst,
              std::memory_order_relaxed)) {
        // LOGA("CircularQueue::pop(): retry since CAS return false.");
        continue;
      }
      v = std::move(data_[tail]);
      clrValid(tail);
      return true;
    }
  }

  uint64_t size() const {
    return n_pushed_.load(std::memory_order_relaxed) -
           n_popped_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<uint64_t>& getAtomicValid(size_t i) {
    return *reinterpret_cast<std::atomic<uint64_t>*>(valid_ + i);
  }

  inline void initValidBits() {
    if constexpr (_use_compact_valid_bits) {
      valid_ = new uint64_t[capacity_ / 64 + 1];
      std::memset(valid_, 0, (capacity_ / 64 + 1) * 8);
    } else {
      ASSERT(capacity_ % 8 == 0);
      valid_ = new uint64_t[capacity_ / 8];
      std::memset(valid_, 0, capacity_);
    }
  }

  inline bool isValid(uint64_t index) {
    if constexpr (_use_compact_valid_bits) {
      std::atomic<uint64_t>& atomic = getAtomicValid(index >> 6);
      uint64_t valid_word = atomic.load(std::memory_order_acquire);
      return valid_word & (1ULL << (index & 63));
    } else {
      std::atomic<uint8_t>& atomic =
          (reinterpret_cast<std::atomic<uint8_t>*>(valid_))[index];
      return atomic.load(std::memory_order_acquire);
    }
  }

  inline void setValid(uint64_t index) {
    if constexpr (_use_compact_valid_bits) {
      std::atomic<uint64_t>& atomic = getAtomicValid(index >> 6);
      uint64_t valid_word = atomic.load(std::memory_order_acquire);
      // while (!atomic.compare_exchange_weak(
      while (!atomic.compare_exchange_strong(
          valid_word, valid_word | (1ULL << (index & 63)),
          std::memory_order_seq_cst, std::memory_order_relaxed)) {
        LOGA("CircularQueue::setValid(): retry since CAS return false.");
      }
    } else {
      std::atomic<uint8_t>& atomic =
          (reinterpret_cast<std::atomic<uint8_t>*>(valid_))[index];
      atomic.store(1, std::memory_order_release);
    }
  }

  inline void clrValid(uint64_t index) {
    if constexpr (_use_compact_valid_bits) {
      std::atomic<uint64_t>& atomic = getAtomicValid(index >> 6);
      uint64_t valid_word = atomic.load(std::memory_order_acquire);
      // while (!atomic.compare_exchange_weak(
      while (!atomic.compare_exchange_strong(
          valid_word, valid_word & (~(1ULL << (index & 63))),
          std::memory_order_seq_cst, std::memory_order_relaxed)) {
        LOGA("CircularQueue::clrValid(): retry since CAS return false.");
      }
    } else {
      std::atomic<uint8_t>& atomic =
          (reinterpret_cast<std::atomic<uint8_t>*>(valid_))[index];
      atomic.store(0, std::memory_order_release);
    }
  }

 private:
  // We add some cacheline paddings so that n_pushed_ and n_popped_ are in
  // different cachelines. Modifying n_pushed_ won't affect the cached
  // n_popped_, and vice versa.
  uint64_t cacheline_padding0[8];
  std::atomic<uint64_t> n_pushed_;

  uint64_t cacheline_padding1[8];
  std::atomic<uint64_t> n_popped_;

  uint64_t cacheline_padding2[8];
  uint64_t capacity_;
  int mod_;
  T* data_;
  uint64_t* valid_;
};

}  // namespace sk4slam
