#pragma once

#include <atomic>   // std::atomic
#include <cstddef>  // offsetof(type, member)
#include <cstdint>  // uint32_t
#include <functional>
#include <limits>  // numeric_limits
#include <memory>
#include <mutex>  // only needed when extending pages.
#include <vector>

#include "sk4slam_basic/align.h"
#include "sk4slam_basic/likely.h"
#include "sk4slam_basic/logging.h"
#include "sk4slam_cpp/thread_pool.h"  // only needed when extending pages in advance.

namespace sk4slam {

using MemPoolSizeType = uint32_t;
using ShortPtrType = uint32_t;
static constexpr ShortPtrType short_nullptr = 0;
inline MemPoolSizeType shortPtr2Entry(const ShortPtrType& short_ptr) {
  return short_ptr - 1;
}
inline ShortPtrType entry2ShortPtr(const MemPoolSizeType& entry) {
  return entry + 1;
}

template <typename PageType>
class ExtenableMemPages;

template <
    typename T, size_t _n_ele_per_page_pow = 8, size_t _short_ptr_bits = 24,
    size_t _aba_tag_bits = 16,
    // bits for 'allocated()' =
    //     64 - (_short_ptr_bits + _aba_tag_bits)
    // If the number of allocated bits is smaller than _short_ptr_bits,
    // then we may need extra atomic counter to trace the number of
    // allocated entries.
    bool _trace_allocated_even_if_need_extra_atomic = true,
    size_t _alignment = (alignof(T) > 8 ? alignof(T) : 8)>
class alignas(_alignment) MemPool final {
 public:
  static_assert(_n_ele_per_page_pow > 0, "n_ele_per_page must be positive");
  static constexpr size_t kNumOfElementsPerPageShift = _n_ele_per_page_pow;
  static constexpr size_t kNumOfElementsPerPage =
      (1ull << kNumOfElementsPerPageShift);

  static_assert(_short_ptr_bits <= 32, "Too many bits for ptr");
  static_assert(_aba_tag_bits <= 32, "Too many bits for aba tag");
  static_assert(
      _short_ptr_bits + _aba_tag_bits <= 64,
      "Too many bits for ptr and aba tag");
  using SizeType = MemPoolSizeType;
  using ThisMemPool = MemPool<
      T, _n_ele_per_page_pow, _short_ptr_bits, _aba_tag_bits,
      _trace_allocated_even_if_need_extra_atomic>;

  static constexpr bool need_extra_allocated_counter =
      (2 * _short_ptr_bits + _aba_tag_bits > 64);
  static constexpr bool has_extra_allocated_counter =
      (need_extra_allocated_counter &&
       _trace_allocated_even_if_need_extra_atomic);
  static constexpr SizeType kPreExtendThr =
      kNumOfElementsPerPage / 10 + 1;  // 0.1 page

  // Thread-safe lock-free ptr. (atomic)
  //     But note that the object pointed by ptr might not be thread safe!
  // Default behavior:
  //   - Changing and copying a shared (visible in many threads) ptr is
  //     thread-safe;
  //   - An rvalue ptr (e.g. std::move()'d) is assumed only visible (or
  //     modifiable) in the current thread so that moving constructor and
  //     moving assignment can be cheaper;
  class Ptr;

  DEFINE_ALIGNED_NEW_OPERATORS(ThisMemPool)
  static std::unique_ptr<ThisMemPool> create(
      SizeType max_capacity = (1 << _short_ptr_bits)) {
    return std::unique_ptr<ThisMemPool>(new ThisMemPool(max_capacity));
  }

  ~MemPool() {
    while (is_pre_extending_.load()) {
      ThreadPool::getNamed(pre_extending_thread_name_)
          ->wait(pre_extending_task_id_);
      LOGA("MemPool::~MemPool(): pre-extending thread done");
    }
    // TODO(jeffrey): Destruct the blocks one by one?
    //     All allocated ptrs should have been deleted (or reset) before we can
    //     destroy the mempool (otherwise it's a user side bug), so it's not
    //     needed to destruct the blocks again.
  }

  inline SizeType capacity() const {
    SizeType capacity = (mem_pages_.n() << kNumOfElementsPerPageShift);
    return capacity > max_capacity_ ? max_capacity_ : capacity;
  }

  inline SizeType allocated() const {
    if constexpr (!need_extra_allocated_counter) {
      // LOGA("MemPool::allocated():  NOT using extra_allocated_counter");
      TaggedShortPtr head = head_.load(std::memory_order_relaxed);
      return head.allocated();
    } else if constexpr (has_extra_allocated_counter) {
      // LOGA("MemPool::allocated():  using extra_allocated_counter");
      return extra_allocated_counter_.load(std::memory_order_relaxed);
    } else {
      LOGE(
          "MemPool::allocated(): _trace_allocated_even_if_need_extra_atomic is "
          "disabled for this class!");
      ASSERT(false);  // issue a bug.
      return 0;
    }
  }

  Ptr alloc() {
  retry_alloc:
    TaggedShortPtr head = head_.load(std::memory_order_relaxed);
    while (UNLIKELY(head.shortptr() == short_nullptr)) {
      // use acquire order to ensure head is not older than mem_pages_.n()
      SizeType npages = mem_pages_.n(std::memory_order_acquire);
      head = head_.load(std::memory_order_acquire);
      if (head.shortptr() == short_nullptr) {
        if (npages < max_pages_) {
          LOGA("MemPool::alloc():  need to extend (cur pages = %zu)", npages);
          extend(npages + 1);
          LOGA("MemPool::alloc():  extend() finished.");
          head = head_.load(std::memory_order_relaxed);
        } else {
          extend(max_pages_);  // sync with pre-extend
          head = head_.load(std::memory_order_relaxed);
          if (head.shortptr() == short_nullptr) {
            return nullptr;
          }
        }
      }
    }
    ASSERT(head.shortptr() != short_nullptr);

    while (UNLIKELY(!head_.compare_exchange_weak(
        head,
        TaggedShortPtr(
            blocks_[shortPtr2Entry(head.shortptr())].next.shortptr(),
            head.allocated() + 1, head.tag() + 1),
        std::memory_order_acq_rel, std::memory_order_relaxed))) {
      if (head.shortptr() == short_nullptr) {
        goto retry_alloc;
      }
    }

    if constexpr (has_extra_allocated_counter) {
      incExtraAllocatedCounter();
    }

    preExtendIfNeeded();

    SizeType entry = shortPtr2Entry(head.shortptr());
    return Ptr(&(blocks_[entry].data));
  }

  inline const T& getConstRef(ShortPtrType short_ptr) const {
    return blocks_[shortPtr2Entry(short_ptr)].data;
  }

  Ptr lock(ShortPtrType short_ptr) {
    SizeType entry = shortPtr2Entry(short_ptr);
    if (short_ptr == short_nullptr || entry >= capacity()) {
      return nullptr;
    }
    return Ptr(&(blocks_[entry].data), false);
  }

  Ptr lock(const T* data) {
    const Block* block = ThisMemPool::Block::getBlock(data);
    const Page* page = getPageByBlock(block);
    if (page->mempool != this) {
      return nullptr;
    }

    // Check for safety.
    ptrdiff_t offset = block - page->blocks;
    ASSERT(block->local_entry == offset);
    auto page_idx = (block->entry >> kNumOfElementsPerPageShift);
    ASSERT(page == mem_pages_.at(page_idx));

    return Ptr(&(blocks_[block->entry].data), false);
  }

 public:
  class alignas(8) Ptr final {
   public:
    DEFINE_ALIGNED_NEW_OPERATORS(Ptr)
    using SizeType = MemPoolSizeType;
    static constexpr auto order_constructing = std::memory_order_relaxed;
    // static constexpr auto order_constructing = std::memory_order_release;

    Ptr() {
      data_.store(nullptr, order_constructing);
    }
    Ptr(std::nullptr_t) {  // NOLINT
      data_.store(nullptr, order_constructing);
    }
    Ptr(const Ptr& other) {
      T* data = other.data_.load(std::memory_order_relaxed);
      while (data != nullptr) {
        typename ThisMemPool::Block* block = ThisMemPool::Block::getBlock(data);
        if (block->incRefCountIfNonZero()) {
          break;
        } else {
          // Ref count for data has been downcount to 0, meaning 'other'
          // has changed during incRefCountIfNonZero(), so try again
          // to copy the updated 'other'.
          // Note a non-null ptr should always have positive ref_count.
          data = other.data_.load(std::memory_order_acquire);
        }
      }
      data_.store(data, order_constructing);
    }

    // It's assumed that 'other' in the move constructor and
    // move assignment is only used by the current thread (at the moment),
    // so weaker atomic operation and memory_order can be used.
    Ptr(Ptr&& other) {
      T* data = other.data_.load(std::memory_order_relaxed);
      other.data_.store(nullptr, std::memory_order_relaxed);
      data_.store(data, order_constructing);
    }

    Ptr& operator=(Ptr&& other) {
      if (this != &other) {
        T* data = other.data_.load(std::memory_order_relaxed);
        other.data_.store(nullptr, std::memory_order_relaxed);
        reset(data, true);
      }
      return *this;
    }

    Ptr& operator=(const Ptr& other) {
      if (this != &other) {
        Ptr copy(other);
        *this = std::move(copy);
      }
      return *this;
    }

    T& operator*() const {
      return *get();
    }
    T* operator->() const {
      return get();
    }
    T* get() const {
      return data_.load(std::memory_order_relaxed);
    }
    bool operator==(const Ptr& other) const {
      return get() == other.get();
    }
    bool operator!=(const Ptr& other) const {
      return get() != other.get();
    }
    bool operator<(const Ptr& other) const {
      return get() < other.get();
    }
    bool operator==(std::nullptr_t) const {
      return get() == nullptr;
    }
    bool operator!=(std::nullptr_t) const {
      return get() != nullptr;
    }
    operator bool() const {
      return get() != nullptr;
    }

    void reset(bool need_thread_safety = true) {
      reset(nullptr, need_thread_safety);
    }

    void become(Ptr&& other, bool need_thread_safety = true) {
      if (!need_thread_safety) {
        if (get() != nullptr) {
          reset(nullptr, false);
        }
        // reconstruct this ptr (this will be very fast).
        new (this) Ptr(std::move(other));
      } else {
        *this = std::move(other);
      }
    }

    ~Ptr() {
      reset(false);
    }

    ShortPtrType shortPtr() const {
      T* data = get();
      if (data) {
        return entry2ShortPtr(ThisMemPool::Block::getBlock(data)->entry);
      } else {
        return short_nullptr;
      }
    }

    SizeType refCount() const {
      T* data = get();
      if (data) {
        auto* blcok = ThisMemPool::Block::getBlock(data);
        return ThisMemPool::getRefCountOfBlock(blcok).load(
            std::memory_order_relaxed);
      } else {
        return 0;
      }
    }

   private:
    Ptr(T* data, bool init = true) {
      if (data == nullptr) {
        data_.store(nullptr, order_constructing);
        return;
      }

      typename ThisMemPool::Block* block = ThisMemPool::Block::getBlock(data);
      if (init) {
        // Construct by alloc()
        ASSERT(block->initRefCount());
        data_.store(data, order_constructing);
      } else {
        // Construct by lock()
        if (block->incRefCountIfNonZero()) {
          data_.store(data, order_constructing);
        } else {
          data_.store(nullptr, order_constructing);
        }
      }
    }

    // mainly needed in operator=().
    void reset(T* other_data, bool need_thread_safety) {
      T* data;
      if (need_thread_safety) {
        data = data_.exchange(other_data, std::memory_order_relaxed);
      } else {
        // faster exchanging, but not atomic.
        // don't call with need_thread_safety=false unless you're sure it's
        // thread-safe.
        data = data_.load(std::memory_order_relaxed);
        data_.store(other_data, std::memory_order_relaxed);
      }

      if (data) {
        typename ThisMemPool::Block* block = ThisMemPool::Block::getBlock(data);
        SizeType entry = block->entry;
        ThisMemPool* pool = block->pool();

        SizeType ref_cnt = block->decRefCount();  // auto memory_order_release
        if (ref_cnt == 0) {
          // Destruct data
          data->~T();

          // release the memory
          pool->free(entry);
        }
        // LOGA("Released entry %d, ref_cnt(after release) = %ld", entry,
        // ref_cnt)
      } else if (need_thread_safety) {
        std::atomic_thread_fence(std::memory_order_release);
      }
    }

    friend class MemPool<
        T, _n_ele_per_page_pow, _short_ptr_bits, _aba_tag_bits,
        _trace_allocated_even_if_need_extra_atomic>;

   private:
    std::atomic<T*> data_;
  };

  class PtrsCache {
   public:
    Ptr& at(SizeType entry) {
      SizeType page_idx = (entry >> kNumOfElementsPerPageShift);
      SizeType npages = ptr_pages_.n();
      if (UNLIKELY(page_idx >= npages)) {
        extend(page_idx + 1);
      } else if (UNLIKELY(
                     (npages << kNumOfElementsPerPageShift) - entry <
                     kPreExtendThr)) {
        // pre-extend in another thread.
        if (!is_pre_extending_.load(std::memory_order_relaxed)) {
          bool tmp = false;
          if (is_pre_extending_.compare_exchange_strong(tmp, true)) {
            pre_extending_task_id_ =
                ThreadPool::getNamed(pre_extending_thread_name_)
                    ->schedule([=]() {
                      LOGA("PtrsCache: pre_extending to %d pages", npages + 1);
                      extend(npages + 1);
                      is_pre_extending_.store(false, std::memory_order_release);
                      LOGA("PtrsCache: pre_extending finished.");
                    });
          }
        }
      }
      SizeType local_entry = (entry & (kNumOfElementsPerPage - 1));
      return ptr_pages_.at(page_idx)->at(local_entry);
    }
    const Ptr& at(SizeType entry) const {
      SizeType page_idx = (entry >> kNumOfElementsPerPageShift);
      ASSERT(page_idx < ptr_pages_.n());
      SizeType local_entry = (entry & (kNumOfElementsPerPage - 1));
      return ptr_pages_.at(page_idx)->at(local_entry);
    }
    Ptr& operator[](SizeType entry) {
      return at(entry);
    }
    const Ptr& operator[](SizeType entry) const {
      return at(entry);
    }

    PtrsCache()
        : is_pre_extending_(false),
          pre_extending_task_id_(INVALID_TASK),
          pre_extending_thread_name_("mempool_ext") {
      extend(1);
    }

    ~PtrsCache() {
      while (is_pre_extending_.load()) {
        ThreadPool::getNamed(pre_extending_thread_name_)
            ->wait(pre_extending_task_id_);
        LOGA("MemPool::~MemPool(): pre-extending thread done");
      }
    }

    SizeType numPages() const {
      return ptr_pages_.n();
    }

   private:
    void extend(SizeType target_num_of_pages) {
      std::unique_lock<std::mutex> lock(mutex_ptr_pages_);
      while (ptr_pages_.n() < target_num_of_pages) {
        ptr_pages_.extendOne([this](SizeType page_idx) {
          return new std::vector<Ptr>(kNumOfElementsPerPage);
        });
      }
    }

   private:
    ExtenableMemPages<std::vector<Ptr>> ptr_pages_;
    std::mutex mutex_ptr_pages_;

    std::atomic<bool> is_pre_extending_;
    TaskID pre_extending_task_id_;
    const char* pre_extending_thread_name_;
  };

  std::shared_ptr<PtrsCache> createPtrsCache() {
    // TODO(jeffrey): maintain pages?
    auto cache = std::make_shared<PtrsCache>();
    auto cap = capacity();
    for (SizeType i = 0; i < cap; ++i) {
      cache->at(i).become(lock(entry2ShortPtr(i)), false);
    }
    return cache;
  }

  // For testing
  SizeType getRefs(ShortPtrType short_ptr) const {
    SizeType entry = shortPtr2Entry(short_ptr);
    auto cap = capacity();
    ASSERT(entry < cap);  // for testing
    if (short_ptr == short_nullptr || entry >= cap) {
      return 0;
    }
    return getRefCountOfBlock(&blocks_[entry]).load(std::memory_order_relaxed);
  }

  SizeType numPages() const {
    return mem_pages_.n();
  }

 private:
  // Note that if max_capacity is (1 << _short_ptr_bits), then the number of
  // the available entries are actually (max_capacity - 1) since
  // the shortptr to the last entry is null, which means it won't be
  // allocated.
  // Otherwise if max_capacity is less than (1 << _short_ptr_bits) and meantime
  // a multiple of kNumOfElementsPerPage, then the available entries are
  // exactly max_capacity.
  explicit MemPool(SizeType max_capacity = (1ull << _short_ptr_bits))
      : max_capacity_(max_capacity),
        extra_allocated_counter_(0),
        blocks_(&mem_pages_),
        is_pre_extending_(false),
        pre_extending_task_id_(INVALID_TASK),
        pre_extending_thread_name_("mempool_ext"),
        head_(TaggedShortPtr()) {
    if ((max_capacity_ & (kNumOfElementsPerPage - 1)) != 0) {
      max_capacity_ = (max_capacity_ & ~(kNumOfElementsPerPage - 1)) +
                      kNumOfElementsPerPage;
      LOGD(
          "MemPool::max_capacity_ %d is not a multiple of "
          "kNumOfElementsPerPage(%d), so it is adjusted to %d",
          max_capacity, kNumOfElementsPerPage, max_capacity_);
    }
    ASSERT((max_capacity_ & (kNumOfElementsPerPage - 1)) == 0);
    if (max_capacity_ > (1ull << _short_ptr_bits)) {
      max_capacity_ = (1ull << _short_ptr_bits);
      LOGD(
          "MemPool::max_capacity_ %d is too large, so it is adjusted to %d",
          max_capacity, max_capacity_);
    }
    ASSERT(max_capacity_ <= (1ull << _short_ptr_bits));
    max_pages_ = (max_capacity_ >> kNumOfElementsPerPageShift);

    // check alignment.
    LOGA(
        "MemPool addr debug:  sizeof(T) = %u, alignof(T) = %u, "
        "sizeof(Block) = %u, alignof(Block) = %u",
        sizeof(T), alignof(T), sizeof(Block), alignof(Block));
    ASSERT(alignof(Block) % alignof(T) == 0);
    // ASSERT(sizeof(Block) % alignof(Block) == 0);

    head_.store(TaggedShortPtr(short_nullptr, 0), std::memory_order_relaxed);
    extend(1);
    if constexpr (has_extra_allocated_counter) {
      extra_allocated_counter_.store(0, std::memory_order_relaxed);
    }
    // head_.store(
    //     TaggedShortPtr(entry2ShortPtr(0), 0),
    //     std::memory_order_release);
  }

  class alignas(8) TaggedShortPtr final {
   public:
    // static constexpr uint32_t shortptr_startbit = 0;
    static constexpr uint32_t tag_startbit = _short_ptr_bits;
    static constexpr uint32_t allocated_startbit =
        _short_ptr_bits + _aba_tag_bits;

    static constexpr uint64_t shortptr_mask = (1ull << _short_ptr_bits) - 1ull;
    static constexpr uint64_t tag_mask = (1ull << _aba_tag_bits) - 1ull;
    static constexpr uint64_t allocated_mask = shortptr_mask;

    constexpr TaggedShortPtr() : v_(0) {}
    constexpr TaggedShortPtr(
        ShortPtrType ptr, uint32_t allocated = 0, uint32_t tag = 0)
        : v_((ptr & shortptr_mask) |
             ((allocated & allocated_mask) << allocated_startbit) |
             ((tag & tag_mask) << tag_startbit)) {}

    uint32_t tag() const {
      return (v_ >> tag_startbit) & tag_mask;
    }
    uint32_t allocated() const {
      return (v_ >> allocated_startbit) & allocated_mask;
    }
    ShortPtrType shortptr() const {
      return (v_ & shortptr_mask);
    }

    DEFINE_ALIGNED_NEW_OPERATORS(TaggedShortPtr)
   private:
    alignas(8) uint64_t v_;

    // ShortPtrType short_ptr_;
    // uint32_t tag_;
  };

  struct alignas(_alignment) Block final {
    T data;
    alignas(8) TaggedShortPtr next;
    SizeType entry;        // global entry in the mempool.
    SizeType local_entry;  // local_entry in the current page.

    Block() {}

    ThisMemPool* pool() {
      return ThisMemPool::getPageByBlock(this)->mempool;
    }

    DEFINE_ALIGNED_NEW_OPERATORS(Block)
    static inline Block* getBlock(T* ptr) {
      // return reinterpret_cast<Block*>(
      //     reinterpret_cast<char*>(ptr) - offsetof(Block, data));
      static_assert(offsetof(Block, data) == 0);
      return reinterpret_cast<Block*>(ptr);
    }
    static inline const Block* getBlock(const T* ptr) {
      // return reinterpret_cast<Block*>(
      //     reinterpret_cast<char*>(ptr) - offsetof(Block, data));
      static_assert(offsetof(Block, data) == 0);
      return reinterpret_cast<const Block*>(ptr);
    }

    // used by alloc()
    // Return 0 for fail; Otherwise return the updated ref count.
    SizeType initRefCount() {
      auto& ref_count = ThisMemPool::getRefCountOfBlock(this);
      ASSERT(ref_count.load(std::memory_order_relaxed) == 0);
      ref_count.store(1, std::memory_order_relaxed);
      return 1;

      // More safe code:
      // clang-format off

      // SizeType ref_cnt = 0;
      // if (ref_count.compare_exchange_strong(
      //         ref_cnt, 1, std::memory_order_acq_rel,
      //         std::memory_order_relaxed)) {
      //   return 1;
      // } else {
      //   return 0;  // failed.
      // }

      // clang-format on
    }

    // Return 0 for fail; Otherwise return the updated ref count.
    SizeType incRefCount() {
      // This function is not used since it might cause overflow of
      // ref_count (0->ffffffff) in multi-thread scene.
      auto& ref_count = ThisMemPool::getRefCountOfBlock(this);
      SizeType old_ref_cnt = ref_count.fetch_add(1, std::memory_order_relaxed);
      ASSERT(old_ref_cnt != 0);
      return old_ref_cnt + 1;
    }

    // used by lock() and copying ptr
    // Return 0 for fail; Otherwise return the updated ref count.
    SizeType incRefCountIfNonZero() {
      auto& ref_count = ThisMemPool::getRefCountOfBlock(this);
      SizeType ref_cnt = ref_count.load(std::memory_order_relaxed);
      if (ref_cnt == 0) {
        return false;
      }
      while (!ref_count.compare_exchange_strong(
          ref_cnt, ref_cnt + 1, std::memory_order_relaxed,
          std::memory_order_relaxed)) {
        if (ref_cnt == 0) {
          return 0;  // failed
        }
      }
      // store CAS result (ref_cnt + 1) to ref_cut.
      ref_cnt = ref_cnt + 1;
      return ref_cnt;
    }

    // when desctructing a Ptr.
    // Return the updated ref count.
    SizeType decRefCount() {
      auto& ref_count = ThisMemPool::getRefCountOfBlock(this);
      SizeType old_ref_cnt = ref_count.fetch_sub(1, std::memory_order_release);
      // ASSERT(old_ref_cnt > 0);
      if (old_ref_cnt == 0) {
        LOGE(
            "Attempting to release an object whose ref_count is already 0! "
            "Might "
            "be a bug!");
        ASSERT(old_ref_cnt > 0);
      }

      if (old_ref_cnt == 1) {
        std::atomic_thread_fence(std::memory_order_acquire);
      }

      return old_ref_cnt - 1;

      // More safe code:
      // clang-format off

      // SizeType ref_cnt = ref_count.load(std::memory_order_relaxed);
      // if (ref_cnt == 0) {
      //   return 0;
      // }
      // while (!ref_count.compare_exchange_strong(
      //     ref_cnt, ref_cnt - 1, std::memory_order_acq_rel,
      //     std::memory_order_relaxed)) {
      //   if (ref_cnt == 0) {
      //     return 0;
      //   }
      // }

      // // store CAS result (ref_cnt - 1) to ref_cut.
      // ref_cnt = ref_cnt - 1;
      // return ref_cnt;

      // clang-format on
    }
  };

  struct alignas(_alignment) Page final {
    Block blocks[kNumOfElementsPerPage];
    alignas(8) SizeType ref_counts[kNumOfElementsPerPage];
    // Why not put every ref_count into the corresponding Block?
    // Since if we do so, the ref_count and the data in the block will belong
    // to (very likely) the same cache line and the cache for data will be
    // invalidated frequently if other threads increase/decrease the ref_count

    ThisMemPool* mempool;
    DEFINE_ALIGNED_NEW_OPERATORS(Page)

    static Page* create(ThisMemPool* mem_pool, SizeType page_idx) {
      void* new_ptr = sk4slam::alignedMalloc<alignof(Page)>(sizeof(Page));
      Page* page = reinterpret_cast<Page*>(new_ptr);
      page->mempool = mem_pool;
      auto& blocks = page->blocks;
      auto& ref_counts = page->ref_counts;
      memset(ref_counts, 0, sizeof(ref_counts));

      SizeType first_entry = page_idx * kNumOfElementsPerPage;
      for (SizeType i = 0; i < kNumOfElementsPerPage; ++i) {
        blocks[i].entry = first_entry + i;
        blocks[i].next = TaggedShortPtr(entry2ShortPtr(blocks[i].entry + 1));
        // Note for the block with "entry = (1 << _short_ptr_bits) - 2",
        // its next shortptr is not pointing to the last block with entry =
        // (1 << _short_ptr_bits) - 1), but pointing to nothing (nullptr).
        // So if max_capacity is (1 << _short_ptr_bits), the number of
        // the available entries are actually  (1 << _short_ptr_bits) - 1.
        blocks[i].local_entry = i;
      }
      blocks[kNumOfElementsPerPage - 1].next = TaggedShortPtr(short_nullptr);

      return page;
    }
  };

  class GetBlockByEntry {
   public:
    explicit GetBlockByEntry(ExtenableMemPages<Page>* pages) : pages_(pages) {}
    inline Block& operator[](SizeType entry) {
      return pages_->at(getPageIdxForEntry(entry))
          ->blocks[getLocalEntry(entry)];
    }
    inline const Block& operator[](SizeType entry) const {
      return pages_->at(getPageIdxForEntry(entry))
          ->blocks[getLocalEntry(entry)];
    }

   private:
    static inline SizeType getPageIdxForEntry(SizeType entry) {
      return entry >> kNumOfElementsPerPageShift;
    }
    static inline SizeType getLocalEntry(SizeType entry) {
      return entry & (kNumOfElementsPerPage - 1);
    }

   private:
    ExtenableMemPages<Page>* pages_;
  };

  void extend(SizeType target_num_of_pages) {
    target_num_of_pages = std::min(target_num_of_pages, max_pages_);
    std::unique_lock<std::mutex> lock(mutex_mem_pages_);
    SizeType prev_n, new_n;
    while ((prev_n = mem_pages_.n()) < target_num_of_pages) {
      mem_pages_.extendOne(
          [this](SizeType page_idx) { return Page::create(this, page_idx); });
      new_n = mem_pages_.n();
      ASSERT(prev_n + 1 == new_n);
      SizeType page_idx = new_n - 1;
      SizeType local_entry = 0;
      SizeType global_entry = getGlobalEntry(page_idx, local_entry);
      SizeType new_head_ptr = entry2ShortPtr(global_entry);
      TaggedShortPtr old_head = head_.load(std::memory_order_relaxed);
      while (!head_.compare_exchange_weak(
          old_head,
          TaggedShortPtr(
              new_head_ptr, old_head.allocated(), old_head.tag() + 1),
          std::memory_order_acq_rel, std::memory_order_relaxed)) {
        // just loop
      }

      // Connect the last entry of the new block with the old head.
      // This is only effective when pre-extending, i.e. calling extend()
      // when there're still free entries (since otherwise
      // the old_head is nullptr and nothing needs to be done).
      //
      // A known corner case:
      // If all the entries of the new page were immediately allocated
      // before we can connect the old head, the head_ pointer will become
      // nullptr again and calls to alloc() in other threads will fail or
      // have to wait. We don't want to fail any alloc() when the pages're
      // not full, so we need to somehow wait for the entries in alloc().
      // An easy method is that whenever the head_ pointer becomes null,
      // just call extend() once and check head_ again.
      //
      mem_pages_.at(page_idx)->blocks[kNumOfElementsPerPage - 1].next =
          old_head;
    }
  }

  void preExtendIfNeeded() {
    static constexpr bool is_allocated_method_available =
        (!need_extra_allocated_counter || has_extra_allocated_counter);

    if constexpr (is_allocated_method_available) {
      SizeType cur_pages = mem_pages_.n();
      SizeType allocated = this->allocated();
      SizeType rest = (cur_pages << kNumOfElementsPerPageShift) - allocated;
      if (LIKELY(rest < kPreExtendThr)) {
        if (LIKELY(is_pre_extending_.load(std::memory_order_relaxed))) {
          return;
        }
        if (UNLIKELY(cur_pages >= max_pages_)) {
          return;
        }
        bool tmp = false;
        if (is_pre_extending_.compare_exchange_strong(tmp, true)) {
          // Run in another thread
          pre_extending_task_id_ =
              ThreadPool::getNamed(pre_extending_thread_name_)->schedule([=]() {
                LOGA(
                    "MemPool: pre_extending to %d pages: current capacity=%d, "
                    "allocated=%d",
                    cur_pages + 1, capacity(), allocated);
                extend(cur_pages + 1);
                is_pre_extending_.store(false, std::memory_order_release);
                LOGA("MemPool: pre_extending finished.");
              });
        } else {
          // Other thread already started pre-extending.
          return;
        }
      }
    }
  }

  static inline const Page* getPageByBlock(const Block* block) {
    static_assert(offsetof(Page, blocks) == 0);
    return reinterpret_cast<const Page*>(block - block->local_entry);
  }
  static inline Page* getPageByBlock(Block* block) {
    return reinterpret_cast<Page*>(block - block->local_entry);
  }
  static inline std::atomic<SizeType>& getRefCountOfBlock(Block* block) {
    return *reinterpret_cast<std::atomic<SizeType>*>(
        getPageByBlock(block)->ref_counts + block->local_entry);
  }
  static inline const std::atomic<SizeType>& getRefCountOfBlock(
      const Block* block) {
    return *reinterpret_cast<const std::atomic<SizeType>*>(
        getPageByBlock(block)->ref_counts + block->local_entry);
  }
  static inline SizeType getGlobalEntry(
      SizeType page_idx, SizeType local_entry) {
    return (page_idx << kNumOfElementsPerPageShift) | local_entry;
  }

  inline void free(SizeType entry) {
    // LOGA("Free entry %d!", entry);
    ASSERT(entry < capacity());
    TaggedShortPtr head = head_.load(std::memory_order_relaxed);
    ShortPtrType new_head_ptr = entry2ShortPtr(entry);

    // Actually ABA problem won't affect free() since now we are inserting
    // new node before the old head.
    TaggedShortPtr& old_head = blocks_[entry].next;
    old_head = head;
    while (UNLIKELY(!head_.compare_exchange_weak(
        old_head,
        TaggedShortPtr(
            new_head_ptr, old_head.allocated() - 1, old_head.tag() + 1),
        std::memory_order_acq_rel, std::memory_order_relaxed))) {
      // just loop
    }

    if constexpr (has_extra_allocated_counter) {
      decExtraAllocatedCounter();
    }
  }

  inline void incExtraAllocatedCounter() {
    // extra_allocated_counter_.fetch_add(1, std::memory_order_acq_rel);
    extra_allocated_counter_.fetch_add(1, std::memory_order_relaxed);
  }

  inline void decExtraAllocatedCounter() {
    // extra_allocated_counter_.fetch_sub(1, std::memory_order_acq_rel);
    extra_allocated_counter_.fetch_sub(1, std::memory_order_relaxed);
  }

  friend class Ptr;

 private:
  alignas(8) std::atomic<SizeType> extra_allocated_counter_;
  alignas(8) std::atomic<TaggedShortPtr> head_;
  SizeType max_capacity_;
  SizeType max_pages_;
  GetBlockByEntry blocks_;

  std::atomic<bool> is_pre_extending_;
  TaskID pre_extending_task_id_;
  const char* pre_extending_thread_name_;

  ExtenableMemPages<Page> mem_pages_;
  std::mutex mutex_mem_pages_;
};

template <typename PageType>
class alignas(8) ExtenableMemPages {
 public:
  using SizeType = MemPoolSizeType;
  // lock-free
  inline PageType* at(
      SizeType i, std::memory_order order = std::memory_order_relaxed
      // std::memory_order_acquire  // <-- safe but slow
  ) const {        // NOLINT
    if (i == 0) {  // the most likely case
      return array_[0];
    } else {
      SizeType n = n_.load(order);
      // SizeType n = n_.load(std::memory_order_acquire);  // <-- safe but slow
      // (ensure sync with array_)
      if (i < n) {
        return array_[i];
      } else {
        ASSERT(i < n);  // bug
        return nullptr;
      }
    }
  }

  // lock-free
  inline SizeType n(std::memory_order order = std::memory_order_relaxed) const {
    return n_.load(order);
  }

  // need lock
  void extendOne(
      const std::function<PageType*(SizeType page_idx)>& new_page = []() {
        return new PageType;
      }) {
    SizeType prev_n = n();
    SizeType new_n = prev_n + 1;
    if (new_n <= reserved_size_) {
      array_[new_n - 1] = new_page(new_n - 1);
      n_.store(new_n, std::memory_order_release);
    } else {
      array_cache_.push_back(array_);
      size_cache_.push_back(reserved_size_);
      while (reserved_size_ < new_n) {
        reserved_size_ *= 2;
      }

      PageType** new_array = new PageType*[reserved_size_];
      memset(new_array, 0, reserved_size_ * sizeof(PageType*));
      memcpy(new_array, array_, prev_n * sizeof(PageType*));
      new_array[new_n - 1] = new_page(new_n - 1);
      array_ = new_array;
      n_.store(new_n, std::memory_order_release);
    }
  }

 public:
  ExtenableMemPages() : reserved_size_(8), n_(0) {
    array_ = new PageType*[reserved_size_];
    memset(array_, 0, reserved_size_ * sizeof(PageType*));
  }

  ~ExtenableMemPages() {
    ASSERT(array_cache_.size() == size_cache_.size());
    for (SizeType i = 0; i < array_cache_.size(); ++i) {
      auto array = array_cache_[i];
      // for (SizeType j = 0; j < size_cache_[i]; ++j) {
      //   delete array[j];
      // }
      delete[] array;
    }

    for (SizeType i = 0; i < n(); ++i) {
      delete array_[i];
    }
    delete[] array_;
  }

 private:
  alignas(8) std::atomic<SizeType> n_;
  alignas(
      8) PageType** array_;  // array_ (as a pointer) can be modified atomicly.

  std::mutex mutex_array_;
  SizeType reserved_size_;
  std::vector<PageType**> array_cache_;
  std::vector<SizeType> size_cache_;
};

}  // namespace sk4slam
