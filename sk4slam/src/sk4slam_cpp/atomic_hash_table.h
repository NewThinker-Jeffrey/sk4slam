#pragma once

#include <functional>
#include <vector>

#include "sk4slam_basic/template_helper.h"
#include "sk4slam_cpp/atomic_list_based_set.h"
#include "sk4slam_cpp/thread_pool.h"  // only needed when rehashing.

// clang-format off
// Ref:
//     https://docs.rs/crate/crossbeam/0.2.4/source/hash-and-skip.pdf
//     https://dl.acm.org/doi/10.1145/564870.564881
//     https://www.researchgate.net/publication/221257109_High_performance_dynamic_lock-free_hash_tables_and_list-based_sets    // NOLINT
// clang-format on

namespace sk4slam {

template <typename KeyType>
using HashFuncTypeT = std::function<size_t(const KeyType&)>;

// cpplint may mistake alignas() as a 'function' and complain about the size
// of the 'function'.
#define HashTableAlignas8 alignas(8)

/// @brief A lock-free hash table.
///
/// A lock-free hash table that supports insert, remove, and find operations.
/// The hash table uses a lock-free linked list to store the confllict entries.
///
/// @tparam T         The type of the elements stored in the hash table.
/// @tparam Hash      The hash function used to hash the keys.
/// @tparam _enable_rehash
///                   Whether to enable rehashing.
/// @tparam _mempool_n_ele_per_page_pow
///                   The power of 2 for the number of elements per page in the
///                   memory pool.
/// @tparam _mempool_short_ptr_bits
///                   The number of bits used to store the short pointer.
/// @tparam _mempool_aba_tag_bits
///                   The number of bits used to store the ABA tag.
/// @tparam _critical_conditions_test
///                   Whether to enable critical conditions testing.
///
/// @note When rehashing is enabled (i.e. @c _enable_rehash is true), the
/// HashTable is no longer perfectly lock-free as the rehashing process requires
/// synchronization. However, only access to the bucket being rehashed may be
/// blocked, see
/// @ref Context::rehashOneBucket() for details.

template <
    typename T, typename Hash = std::hash<T>, bool _enable_rehash = false,
    // Parameters for the underlying mempool. See the template parameter
    // list of MemPool.
    size_t _mempool_n_ele_per_page_pow = 8, size_t _mempool_short_ptr_bits = 24,
    size_t _mempool_aba_tag_bits = 16, bool _critical_conditions_test = false>
// Type T is required to support comparing operators (<,==) with itself,
// i.e.  (T1 < T2), (T1 == T2) shuold be legal.
// Also Note that (T1 == T2) should only implies they have the same key, not
// meaning they are totally identical.
// Besides, the 'key' member(s) of type T should be constant, otherwise you
// should be careful to use the ptrs returned by insert() or find() (do not
// change the key of ptr->data).

class HashTableAlignas8 HashTable {
 public:
  using BasedList = ListBasedSet<
      T, _mempool_n_ele_per_page_pow, _mempool_short_ptr_bits,
      _mempool_aba_tag_bits, _critical_conditions_test>;
  using NodeType = typename BasedList::NodeType;
  using MarkPtrType = typename BasedList::MarkPtrType;
  using MemoryManager = typename BasedList::MemoryManager;
  using InsertReturnType = typename BasedList::InsertReturnType;
  using UnderlyingMemPool = typename BasedList::UnderlyingMemPool;
  using NodePtr = typename BasedList::NodePtr;
  using HashFuncType = HashFuncTypeT<T>;

  // See "Figure 3: Hash table operations" in the ref above.

  explicit HashTable(
      size_t n_buckets = 1024, const Hash& hash = Hash(),
      size_t max_capacity = (1ull << _mempool_short_ptr_bits))
      : first_context_(new Context(n_buckets, max_capacity)),
        rehashing_task_id_(INVALID_TASK),
        rehashing_thread_name_("rehashing"),
        hash_(hash) {
    if constexpr (_enable_rehash) {
      cur_context_.store(first_context_);
      is_rehashing_.store(false);
    }
  }

  // Copy is not allowed
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;
  HashTable& operator=(HashTable&&) = delete;

  // Move is allowed (but not thread-safe)
  HashTable(HashTable&& other)
      : hash_(std::move(other.hash_)),
        rehashing_task_id_(INVALID_TASK),
        rehashing_thread_name_("rehashing"),
        first_context_(other.first_context_) {
    other.first_context_ = nullptr;
    if constexpr (_enable_rehash) {
      cur_context_.store(other.cur_context_.load());
      other.cur_context_.store(nullptr);
    }
  }

  ~HashTable() {
    if constexpr (_enable_rehash) {
      LOGA("~HashTable: waitForRehashing");
      waitForRehashing();
      LOGA("~HashTable: deleteAllContexts");
    }
    deleteAllContexts();
    LOGA("~HashTable: done");
  }

  // Known flaw:
  //   size() actually reflects the number of nodes allocated, which might be
  //   larger than the real size of the table during some special stages.
  //   To be more specific:
  //        real size <= size() <= (real size + K + M)
  //   where K is number of concurrent calls to insert() while M the number of
  //   erased nodes which are still referenced (temporarily) by external code.
  size_t size() const {
    // Note that all contexts share the same mempool, so we just get the mempool
    // from the first context.
    return first_context_->mm.allocated();
  }

  // memory usage
  size_t capacity() const {
    return first_context_->mm.capacity();
  }

  size_t bucketCount() const {
    return getCurrentContext()->mod + 1;
  }

  // Always sucess, but it might be slow since all the entries in the
  // underlying mem_pool_ (See MemoryManager) will be checked.
  void foreachAlongMemory(  // NOLINT
      const std::function<void(T&)>& func) {
    // getCurrentContext()->mm.foreachAlongMemory(func);
    // Note that all contexts share the same mempool, so we just get the
    // capacity from the first context. Also note that the capacity might be
    // extended during running foreach().
    for (size_t i = 0; i < first_context_->mm.capacity(); ++i) {
      NodePtr ptr = getCurrentContext()->mm.getKeptNode(entry2ShortPtr(i));
      if (ptr != nullptr) {
        func(ptr->data);
      }
    }
  }
  void foreachAlongMemory(  // NOLINT
      const std::function<void(const T&)>& func) const {
    // getCurrentContext()->mm.foreachAlongMemory(func);
    for (size_t i = 0; i < first_context_->mm.capacity(); ++i) {
      NodePtr ptr = getCurrentContext()->mm.getKeptNode(entry2ShortPtr(i));
      if (ptr != nullptr) {
        func(ptr->data);
      }
    }
  }

  // wait for rehashing over
  void waitForRehashing() {
    if constexpr (_enable_rehash) {
      while (is_rehashing_.load()) {
        ThreadPool::getNamed(rehashing_thread_name_)->wait(rehashing_task_id_);
        LOGA("HashTable::waitForRehashing(): rehashing thread done");
      }
    }
  }

 public:
  template <typename... Args>
  inline InsertReturnType insert(Args&&... args) {
    Context* c = getCurrentContext();
    ShortPtrType new_short_ptr;
    auto new_node = c->mm.newNode(&new_short_ptr);
    if (!new_node) {
      LOGA("HashTable::insert(): Fail to alloc node!");
      return InsertReturnType(false, nullptr);
    }
    // use placement new to construct at new_node->data.
    new (&new_node->data) T(std::forward<Args>(args)...);

    if constexpr (_enable_rehash) {
      if (size() >= (c->mod + 1) * 3 / 4) {
        rehash(c);
      }
      return recursiveInsert(
          c, hash(new_node->data), std::move(new_node), new_short_ptr);
    } else {
      int bucket_idx = c->getBucketIdx(hash(new_node->data));
      return c->insert(std::move(new_node), new_short_ptr, bucket_idx);
    }
  }

  // find() and erase() can be used for short keytypes (If Hash class has
  // overloaded operator() for short keytypes).
  // Type T is required to support comparing operators (<,==) with KeyType,
  // i.e.  (T < KeyType), (T == KeyType) shuold be legal.
  template <typename KeyType>
  inline bool erase(const KeyType& key) {
    Context* c = getCurrentContext();
    if constexpr (_enable_rehash) {
      return recursiveErase(c, hash(key), key);
    } else {
      int bucket_idx = c->getBucketIdx(hash(key));
      return c->erase(key, bucket_idx);
    }
  }

  template <typename KeyType>
  inline NodePtr find(const KeyType& key) const {
    Context* c = getCurrentContext();
    if constexpr (_enable_rehash) {
      return recursiveFind(c, hash(key), key);
    } else {
      int bucket_idx = c->getBucketIdx(hash(key));
      return c->find(key, bucket_idx);
    }
  }

 protected:
  template <typename KeyType>
  inline size_t hash(const KeyType& key) const {
    return hash_(key);
  }

  struct Context;
  void rehash(Context* c) {
    if (is_rehashing_.load(std::memory_order_relaxed)) {
      return;
    }
    bool tmp = false;
    if (is_rehashing_.compare_exchange_strong(tmp, true)) {
      rehashing_task_id_ =
          ThreadPool::getNamed(rehashing_thread_name_)->schedule([=]() {
            // Context* c = cur_context_.load(std::memory_order_acquire);
            LOGA("HashTable: rehashing to %d buckets ...", (c->mod + 1) << 1);
            if (c->rehash(hash_)) {
              cur_context_.store(c->next, std::memory_order_release);
              LOGA(
                  "HashTable: finished rehashing to %d buckets.", (c->mod + 1)
                                                                      << 1);
            } else {
              LOGA(
                  "HashTable: replicate rehash operation (to %d buckets).",
                  (c->mod + 1) << 1);
            }
            is_rehashing_.store(false, std::memory_order_release);
          });
    }
    // waitForRehashing();
  }

  inline InsertReturnType recursiveInsert(
      Context* c, size_t hash_v, NodePtr&& new_node,
      ShortPtrType new_short_ptr) {
    int rehashing_idx, bucket_idx;
    while ((rehashing_idx = c->rehashing_idx.load(std::memory_order_acquire)) >=
           (bucket_idx = c->getBucketIdx(hash_v))) {
      if (rehashing_idx == bucket_idx) {
        c->rehashOneBucket(
            bucket_idx, hash_);  // sync with the rehashing thread
      }
      c = c->next;
    }
    ASSERT(rehashing_idx < bucket_idx);

    T* data = &new_node->data;
    NodePtr forward_fail_node;
    auto ret = c->insert(
        std::move(new_node), new_short_ptr, bucket_idx, &forward_fail_node);

    // Recheck if the bucket has been rehashed.
    rehashing_idx = c->rehashing_idx.load(std::memory_order_acquire);
    if (rehashing_idx < bucket_idx) {
      return ret;
    } else {
      if (rehashing_idx == bucket_idx) {
        c->rehashOneBucket(
            bucket_idx, hash_);  // sync with the rehashing thread
      }
      if (ret.first) {
        // clang-format off
        LOGA(
            "The bucket has been rehashed during Insert().case A:"
            " bucket_idx = %d, mod + 1 = %d",
            bucket_idx, c->mod + 1);
        // clang-format on

        // clang-format off
        c->mm.deleteKeptNode(
            new_short_ptr);  // make sure the the new node is removed
                             // from the c->mm since it's already kept
                             // in c->next->mm after rehash.
        // clang-format on

        return ret;
      } else {
        // clang-format off
        LOGA(
            "The bucket has been rehashed during Insert().case B:"
            " bucket_idx = %d, mod + 1 = %d",
            bucket_idx, c->mod + 1);
        // clang-format on
        ASSERT(forward_fail_node);
        ASSERT(&forward_fail_node->data == data);
        return recursiveInsert(
            c->next, hash_v, std::move(forward_fail_node), new_short_ptr);
      }
    }
  }

  // find() and erase() can be used for short keytypes (If Hash class has
  // overloaded operator() for short keytypes).
  // Type T is required to support comparing operators (<,==) with KeyType,
  // i.e.  (T < KeyType), (T == KeyType) shuold be legal.
  template <typename KeyType>
  inline bool recursiveErase(Context* c, size_t hash_v, const KeyType& key) {
    int rehashing_idx, bucket_idx;
    while ((rehashing_idx = c->rehashing_idx.load(std::memory_order_acquire)) >=
           (bucket_idx = c->getBucketIdx(hash_v))) {
      if (rehashing_idx == bucket_idx) {
        c->rehashOneBucket(
            bucket_idx, hash_);  // sync with the rehashing thread
      }
      c = c->next;
    }
    ASSERT(rehashing_idx < bucket_idx);

    auto ret = c->erase(key, bucket_idx);
    // Recheck if the bucket has been rehashed.
    rehashing_idx = c->rehashing_idx.load(std::memory_order_acquire);
    if (rehashing_idx < bucket_idx) {
      return ret;
    } else {
      if (rehashing_idx == bucket_idx) {
        c->rehashOneBucket(bucket_idx, hash_);
      }
      if (ret) {
        // clang-format off
        LOGA(
            "The bucket has been rehashed during Erase().case A:"
            " bucket_idx = %d, mod + 1 = %d",
            bucket_idx, c->mod + 1);
        // clang-format on
        return ret;
      } else {
        // clang-format off
        LOGA(
            "The bucket has been rehashed during Erase().case B:"
            " bucket_idx = %d, mod + 1 = %d",
            bucket_idx, c->mod + 1);
        // clang-format on
        return recursiveErase(c->next, hash_v, key);
      }
    }
  }

  template <typename KeyType>
  inline NodePtr recursiveFind(
      Context* c, size_t hash_v, const KeyType& key) const {
    int rehashing_idx, bucket_idx;
    while ((rehashing_idx = c->rehashing_idx.load(std::memory_order_acquire)) >=
           (bucket_idx = c->getBucketIdx(hash_v))) {
      if (rehashing_idx == bucket_idx) {
        c->rehashOneBucket(
            bucket_idx, hash_);  // sync with the rehashing thread
      }
      c = c->next;
    }
    ASSERT(rehashing_idx < bucket_idx);

    auto ret = c->find(key, bucket_idx);
    // Recheck if the bucket has been rehashed.
    rehashing_idx = c->rehashing_idx.load(std::memory_order_acquire);
    if (rehashing_idx < bucket_idx) {
      return ret;
    } else {
      // clang-format off
      LOGA(
          "The bucket has been rehashed during Find():"
          " bucket_idx = %d, mod + 1 = %d",
          bucket_idx, c->mod + 1);
      // clang-format on
      if (rehashing_idx == bucket_idx) {
        c->rehashOneBucket(bucket_idx, hash_);
      }
      return recursiveFind(c->next, hash_v, key);
    }
  }

 protected:
  struct Context {
    std::atomic<MarkPtrType>* buckets;
    mutable MemoryManager mm;
    const int mod;
    std::atomic<int> rehashing_idx;
    Context* next;

    Context(size_t n_buckets, size_t max_capacity)
        : mm(max_capacity),
          mod(n_buckets - 1),
          rehashing_idx(-1),
          next(nullptr) {
      initBuckets(n_buckets);
    }

    ~Context() {
      if (buckets != nullptr) {
        delete[] buckets;
      }

      delete next;  // recursively destruct the list might cause stack overflow,
                    // but this rarely happen for our case.
    }

    inline int getBucketIdx(size_t hash_v) {
      return (hash_v & mod);
    }

    inline InsertReturnType insert(
        NodePtr&& new_node, ShortPtrType new_short_ptr, int bucket_idx,
        NodePtr* forward_fail_node = nullptr) {
      std::atomic<MarkPtrType>* bucket_head = buckets + bucket_idx;
      return BasedList::insert(
          std::move(new_node), new_short_ptr, bucket_head, &mm,
          forward_fail_node);
    }

    template <typename KeyType>
    inline bool erase(const KeyType& key, int bucket_idx) {
      std::atomic<MarkPtrType>* bucket_head = buckets + bucket_idx;
      return BasedList::erase(key, bucket_head, &mm);
    }

    template <typename KeyType>
    inline NodePtr find(const KeyType& key, int bucket_idx) const {
      std::atomic<MarkPtrType>* bucket_head = buckets + bucket_idx;
      return BasedList::find(key, bucket_head, &mm);
    }

    bool rehash(const Hash& new_hash) {
      if (rehashing_idx.load() > 0) {
        return false;
      }
      next = new Context((mod + 1) << 1, mm.getUnderlyingMemPool());
      rehashing_idx.store(0, std::memory_order_release);
      for (size_t i = 0; i < mod + 1; ++i) {
        rehashOneBucket(i, new_hash);
      }
      ASSERT(rehashing_idx.load(std::memory_order_relaxed) == mod + 1);
      return true;
    }

    void initBuckets(size_t n_buckets) {
      // clang-format off
      // buckets = new std::atomic<MarkPtrType>[n_buckets];
      //           ^
      // error: use of deleted function ‘std::atomic<_Tp>::atomic()
      // [with _Tp = sk4slam::ListBasedSet<ov_msckf::dense_mapping::CubeBlock>::MarkPtrType]’  // NOLINT
      // clang-format on
      buckets = reinterpret_cast<std::atomic<MarkPtrType>*>(
          new MarkPtrType[n_buckets]);
      BasedList::MarkPtrType::initArray(
          reinterpret_cast<MarkPtrType*>(buckets), n_buckets);
    }

    void rehashOneBucket(int bucket_idx, const Hash& new_hash) {
      if (rehashing_idx.load(std::memory_order_acquire) != bucket_idx) {
        return;
      }

      // TODO(jeffrey): Is it possible to avoid spinning ?
      auto wait = [&]() {
        // spin
        const int holdon_loops = 64;
        int loops = holdon_loops;
        while (rehashing_idx.load(std::memory_order_acquire) == bucket_idx) {
          if ((--loops) == 0) {
            loops = holdon_loops;
            std::this_thread::yield();
          }
        }
      };

      std::atomic<MarkPtrType>* bucket = buckets + bucket_idx;
      MarkPtrType tmpp = bucket->load(std::memory_order_acquire);

      bool is_first_call = false;
      while (!tmpp.mark()) {
        if (bucket->compare_exchange_strong(
                tmpp, MarkPtrType(1, tmpp.next(), tmpp.tag() + 1),
                std::memory_order_release, std::memory_order_acquire)) {
          is_first_call = true;
        }
      }

      if (!is_first_call) {
        wait();
        return;
      }

      while (true) {
        // if constexpr (_critical_conditions_test) {
        //   std::this_thread::sleep_for(std::chrono::microseconds(1));
        // }

        tmpp = bucket->load(std::memory_order_acquire);
        ASSERT(tmpp.mark());
        auto cur = tmpp.next();
        if (cur == short_nullptr) {
          break;  // finish
        }

        NodePtr cur_node = mm.getNode(cur);
        ASSERT(cur_node != nullptr);

        // clang-format off
        if (mm.getKeptNode(cur) == nullptr) {
          LOGA(
              "rehashOneBucket %d:  the node %lx (with short_ptr %d) "
              "has not been kept by the mm yet, and it's going to be "
              "kept in the thread that is inserting the node. but it's "
              "ok since we'll ensure the node been removed from the mm "
              "later in recursiveInsert()",
              bucket_idx, cur_node.get(), cur);
          // continue;  // <-- not necessary.
        }
        // clang-format on

        // if constexpr (_critical_conditions_test) {
        //   std::this_thread::sleep_for(std::chrono::microseconds(1));
        // }

        auto c = cur_node->mark_ptr.load(std::memory_order_acquire);
        if (!c.mark()) {
          if (cur_node->mark_ptr.compare_exchange_strong(
                  c, MarkPtrType(1, c.next(), c.tag() + 1),
                  std::memory_order_release, std::memory_order_acquire)) {
            // if constexpr (_critical_conditions_test) {
            //   std::this_thread::sleep_for(std::chrono::microseconds(1));
            // }

            int bucket_idx_in_next =
                next->getBucketIdx(new_hash(cur_node->data));
            auto insert_res = next->insert(
                NodePtr(cur_node), cur_node.shortPtr(), bucket_idx_in_next);

            // clang-format off
            // LOGA(
            //     "rehashOneBucket %d:  bucket_idx_in_next %d, insert_res %d  "
            //     "%lx",
            //     bucket_idx, bucket_idx_in_next, insert_res.first,
            //     insert_res.second.get());
            // clang-format on

            // ASSERT(insert_res.first);
            ASSERT(insert_res.second == cur_node);
          } else {
            // clang-format off
            LOGA(
                "Some concurrent modification was made to the bucket"
                " %d during rehashing, mod + 1 = %d",
                bucket_idx, mod + 1);
            // clang-format on
            continue;
          }
        }

        // if constexpr (_critical_conditions_test) {
        //   std::this_thread::sleep_for(std::chrono::microseconds(1));
        // }

        mm.deleteKeptNode(cur);

        // if constexpr (_critical_conditions_test) {
        //   std::this_thread::sleep_for(std::chrono::microseconds(1));
        // }

        // CAS might return false if other thread did it earlier, but it's ok.
        ASSERT(bucket->compare_exchange_strong(
            tmpp, MarkPtrType(1, c.next(), tmpp.tag() + 1),
            std::memory_order_release, std::memory_order_acquire));
      }

      // if constexpr (_critical_conditions_test) {
      //   std::this_thread::sleep_for(std::chrono::microseconds(1));
      // }

      rehashing_idx.compare_exchange_strong(
          bucket_idx, bucket_idx + 1, std::memory_order_release,
          std::memory_order_acquire);
    }

   private:
    Context(size_t n_buckets, std::shared_ptr<UnderlyingMemPool> mempool)
        : mm(mempool), mod(n_buckets - 1), rehashing_idx(-1), next(nullptr) {
      initBuckets(n_buckets);
    }

   public:
    // for test
    size_t countElements() {
      size_t total = 0;
      for (size_t i = 0; i < mod + 1; i++) {
        total += BasedList::snapshot(buckets + i, &mm, false).size();
      }
      return total;
    }
  };

  void deleteAllContexts() {
    delete first_context_;  // all subsequent contexts will be deleted
                            // recursively.
    first_context_ = nullptr;
  }

  inline Context* getCurrentContext(
      std::memory_order order = std::memory_order_relaxed) const {
    if constexpr (_enable_rehash) {
      return cur_context_.load(order);
    } else {
      return first_context_;
    }
  }

 public:
  // for test.
  bool verifyRehashing() {
    if constexpr (_enable_rehash) {
      // ensure rehashing is over
      ASSERT(is_rehashing_.load() == false);

      Context* c = cur_context_.load();

      // the new table should be rehashing
      ASSERT(c->rehashing_idx.load() < 0);

      // // ensure the first_context_ already been rehased.
      // ASSERT(c !=  first_context_);

      Context* to_test = first_context_;
      while (to_test != c) {
        ASSERT(to_test->rehashing_idx.load() == (to_test->mod + 1));

        // The old tables should be empty
        for (int i = 0; i < to_test->mod + 1; i++) {
          ASSERT(
              BasedList::snapshot(to_test->buckets + i, &(to_test->mm), false)
                  .empty());
        }

        // The old tables shuoldn't keep nodes
        LOGA("to_test->mm.numKeptNodes() = %d", to_test->mm.numKeptNodes());
        ASSERT(to_test->mm.numKeptNodes() == 0);
        to_test = to_test->next;
      }
      return true;
    } else {
      return true;
    }
  }

  // for test
  int getCurrentRehasingIdx() {
    return getCurrentContext()->rehashing_idx.load();
  }

  // for test
  void foreachAlongBuckets(  // NOLINT
      const std::function<void(size_t bucket_idx, const T&)>& func) {
    Context* c = getCurrentContext();
    for (size_t i = 0; i < c->mod + 1; i++) {
      ASSERT(BasedList::foreach (  // NOLINT
          [i, &func](const NodePtr& node) { func(i, node->data); },
          c->buckets + i, &(c->mm)));
    }
  }

  // for test
  size_t countElements() {
    Context* c = getCurrentContext();
    return c->countElements();
  }

 private:
  alignas(8) std::atomic<Context*> cur_context_;
  Context* first_context_;
  Hash hash_;

  std::atomic<bool> is_rehashing_;
  TaskID rehashing_task_id_;
  const char* rehashing_thread_name_;
};

template <typename KeyType, typename ValueType>
struct KeyValuePairT {
  const KeyType key;
  ValueType value;

  inline bool operator==(const KeyValuePairT& other) const {
    return key == other.key;
  }
  inline bool operator<(const KeyValuePairT& other) const {
    return key < other.key;
  }
  inline bool operator==(const KeyType& other_key) const {
    return key == other_key;
  }
  inline bool operator<(const KeyType& other_key) const {
    return key < other_key;
  }

  template <typename KeyTypeForward, typename... ValueArgs>
  KeyValuePairT(KeyTypeForward&& k, ValueArgs&&... vargs)  // NOLINT
      : key(std::forward<KeyTypeForward>(k)),
        value(std::forward<ValueArgs>(vargs)...) {}

  template <typename KeyTypeForward>
  explicit KeyValuePairT(KeyTypeForward&& k)
      : key(std::forward<KeyTypeForward>(k)) {}

  KeyValuePairT() : key(KeyType()) {}
};

template <typename KeyType, typename ValueType, typename Hash>
struct PairHashT : public Hash {
  template <typename... Args>
  PairHashT(Args&&... args) : Hash(std::forward<Args>(args)...) {  // NOLINT
    LOGA("Constructing PairHashT with args");
  }

  PairHashT() : Hash() {}

  inline size_t operator()(const KeyValuePairT<KeyType, ValueType>& kv) const {
    return Hash::operator()(kv.key);
  }
};

template <
    typename KeyType, typename ValueType, typename Hash = std::hash<KeyType>,
    bool _enable_rehash = false,
    // Parameters for the underlying mempool. See the template parameter
    // list of MemPool.
    size_t _mempool_n_ele_per_page_pow = 8, size_t _mempool_short_ptr_bits = 24,
    size_t _mempool_aba_tag_bits = 16, bool _critical_conditions_test = false>
using HashMap = HashTable<
    KeyValuePairT<KeyType, ValueType>, PairHashT<KeyType, ValueType, Hash>,
    _enable_rehash, _mempool_n_ele_per_page_pow, _mempool_short_ptr_bits,
    _mempool_aba_tag_bits>;

///// unit tests ////
template <
    typename T, typename Hash = std::hash<T>, bool _enable_rehash = false,
    size_t _mempool_n_ele_per_page_pow = 8, size_t _mempool_short_ptr_bits = 24,
    size_t _mempool_aba_tag_bits = 16>
using HashTableForUnitTest = HashTable<
    T, Hash, _enable_rehash, _mempool_n_ele_per_page_pow,
    _mempool_short_ptr_bits, _mempool_aba_tag_bits, true>;

template <
    typename KeyType, typename ValueType, typename Hash = std::hash<KeyType>,
    bool _enable_rehash = false, size_t _mempool_n_ele_per_page_pow = 8,
    size_t _mempool_short_ptr_bits = 24, size_t _mempool_aba_tag_bits = 16>
using HashMapForUnitTest = HashMap<
    KeyType, ValueType, Hash, _enable_rehash, _mempool_n_ele_per_page_pow,
    _mempool_short_ptr_bits, _mempool_aba_tag_bits, true>;

}  // namespace sk4slam
