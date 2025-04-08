#pragma once

#include <chrono>
#include <functional>
#include <thread>

#include "sk4slam_cpp/mem_pool.h"

// clang-format off
// Ref:
//     https://docs.rs/crate/crossbeam/0.2.4/source/hash-and-skip.pdf
//     https://dl.acm.org/doi/10.1145/564870.564881
//     https://www.researchgate.net/publication/221257109_High_performance_dynamic_lock-free_hash_tables_and_list-based_sets    // NOLINT
// clang-format on

namespace sk4slam {

template <
    typename T,
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
class ListBasedSet final {
 public:
  using ABATagType = uint32_t;
  struct MarkPtrType;
  struct NodeType;
  using UnderlyingMemPool = MemPool<
      NodeType, _mempool_n_ele_per_page_pow, _mempool_short_ptr_bits,
      _mempool_aba_tag_bits, _critical_conditions_test>;
  using NodePtr = typename UnderlyingMemPool::Ptr;

  using InsertReturnType = std::pair<bool, NodePtr>;
  // clang-format off
      // Explanation to InsertReturnType:
      //    ----------------------------
      //    ----------------------------
      //    first |   second   | discription
      //    ----------------------------
      //    ----------------------------
      //     true |  non-null  | New node has been inserted successfully (.second points to the new node)                         | // NOLINT
      //    ----------------------------                                                                                          | // NOLINT
      //    false |  non-null  | An earlier node with the same key was already inserted [already visible to find() and erase()]   | // NOLINT
      //                       | or just begined its insert() but still not finished [not visible to find() or erase() yet].      | // NOLINT
      //                       | (.second points to the earlier node)                                                             | // NOLINT
      //    ----------------------------                                                                                          | // NOLINT
      //    false |    null    | The list head is marked or the memory pool is full                                               | // NOLINT
      //    ----------------------------
  // clang-format on

  // Default max capacity is (2^24 = 16M) (See MemPool)
  ListBasedSet(size_t max_capacity = (1ull << _mempool_short_ptr_bits))
      : mm_(max_capacity) {}

  // Copy is not allowed
  ListBasedSet(const ListBasedSet&) = delete;
  ListBasedSet& operator=(const ListBasedSet&) = delete;
  ListBasedSet& operator=(ListBasedSet&&) = delete;

  // Move is allowed but not thread-safe
  ListBasedSet(ListBasedSet&&) = default;

  // Known flaw:
  //   size() actually reflects the number of nodes allocated, which might be
  //   larger than the real size of the list during some special stages.
  //   To be more specific:
  //        real size <= size() <= (real size + K + M)
  //   where K is number of concurrent calls to insert() while M the number of
  //   erased nodes which are still being referenced (temporarily, e.g. in
  //   foreach())
  size_t size() const {
    return mm_.allocated();
  }

  template <typename... Args>
  inline InsertReturnType insert(Args&&... args) {
    ShortPtrType new_short_ptr;
    auto new_node = mm_.newNode(&new_short_ptr);
    if (!new_node) {
      LOGA("ListBasedSet::insert(): Fail to alloc node!");
      return InsertReturnType(false, nullptr);
    }
    // use placement new to construct at new_node->data.
    new (&new_node->data) T(std::forward<Args>(args)...);
    return insert(std::move(new_node), new_short_ptr, &head_, &mm_);
  }

  template <typename KeyType>
  // Type T is required to support comparing operators (<,==) with KeyType,
  // i.e.  (T < KeyType), (T == KeyType) shuold be legal.
  inline bool erase(const KeyType& key) {
    return erase(key, &head_, &mm_);
  }

  template <typename KeyType>
  // Type T is required to support comparing operators (<,==) with KeyType,
  // i.e.  (T < KeyType), (T == KeyType) shuold be legal.
  inline NodePtr find(const KeyType& key) const {
    return find(key, &head_, &mm_);
  }

  // May fail (return false) if there're new changes to the list that happen to
  // insert new node (or erase the node) just before the next node to process.
  bool foreach (const std::function<void(T&)>& f) {  // NOLINT
    return foreach ([&](const NodePtr& p) { f(p->data); }, &head_, &mm_);
  }

  bool foreach (const std::function<void(const T&)>& f) const {  // NOLINT
    return foreach ([&](const NodePtr& p) { f(p->data); }, &head_, &mm_);
  }

  // Always sucess, but it needs to keep extra references to the nodes until
  // the corresponding job done
  void foreachUsingSnapshot(const std::function<void(T&)>& f) {
    std::vector<NodePtr> nodes = snapshot(&head_, &mm_);
    for (const auto& node : nodes) {
      f(node->data);
    }
  }

  void foreachUsingSnapshot(const std::function<void(const T&)>& f) const {
    std::vector<NodePtr> nodes = snapshot(&head_, &mm_);
    for (const auto& node : nodes) {
      f(node->data);
    }
  }

  // Always sucess, but it might be slow since all the entries in the
  // underlying mem_pool_ (See MemoryManager) will be checked.
  void foreachAlongMemory(const std::function<void(T&)>& f) {
    mm_.foreachAlongMemory(f);
  }

  void foreachAlongMemory(const std::function<void(const T&)>& f) const {
    mm_.foreachAlongMemory(f);
  }

 public:
  // See "Figure 2: Types and structures" in the ref above.
  struct alignas(8) MarkPtrType {
    static void initArray(MarkPtrType* array, size_t n = 1) {
      std::memset(array, 0, n * sizeof(MarkPtrType));
    }
    MarkPtrType() {
      u_.v = 0;
    }
    MarkPtrType(bool mark, ShortPtrType next, ABATagType tag) {
      u_.next = next;
      u_.mark_and_tag = (tag & kTagMask) | (mark ? kMarkMask : 0);
    }

    inline ShortPtrType next() const {
      return u_.next;
    }

    // mark() == 1 means the node is going to be removed (shouldn't
    // be used anymore).
    inline bool mark() const {
      return u_.mark_and_tag & kMarkMask;
    }

    inline ABATagType tag() const {
      return u_.mark_and_tag & kTagMask;
    }

    inline void setTag(ABATagType tag) {
      u_.mark_and_tag = (u_.mark_and_tag & kMarkMask) | (tag & kTagMask);
    }

    inline bool operator==(const MarkPtrType& other) const {
      static_assert(sizeof(MarkPtrType) == 8, "MarkPtrType should be 64-bit");
      return u_.v == other.u_.v;
    }

    inline bool operator!=(const MarkPtrType& other) const {
      static_assert(sizeof(MarkPtrType) == 8, "MarkPtrType should be 64-bit");
      return u_.v != other.u_.v;
    }

   private:
    union {
      struct {
        ShortPtrType next;
        ABATagType mark_and_tag;
      };
      uint64_t v;  // force the union 8 bytes aligned.
    } u_;
    static constexpr ABATagType kMarkMask = 1 << 31;
    static constexpr ABATagType kTagMask = ~kMarkMask;
  };

  struct alignas(8) NodeType {
    T data;

    NodeType() : mark_ptr(MarkPtrType()) {}

   private:
    alignas(8) std::atomic<MarkPtrType> mark_ptr;

    friend class ListBasedSet<
        T, _mempool_n_ele_per_page_pow, _mempool_short_ptr_bits,
        _mempool_aba_tag_bits, _critical_conditions_test>;

    friend class FindResult;

    template <
        typename tT, typename tHash, bool t_enable_rehash,
        size_t t_mempool_n_ele_per_page_pow, size_t t_mempool_short_ptr_bits,
        size_t t_mempool_aba_tag_bits, bool t_critical_conditions_test>
    friend class HashTable;

    static const NodeType* getNodeContaining(
        const std::atomic<MarkPtrType>* p_mark_ptr) {
      return reinterpret_cast<const NodeType*>(
          reinterpret_cast<const char*>(p_mark_ptr) -
          offsetof(NodeType, mark_ptr));
    }
  };

  class MemoryManager {
   public:
    explicit MemoryManager(size_t max_capacity)
        : mem_pool_(UnderlyingMemPool::create(max_capacity)),
          node_pool_(new typename UnderlyingMemPool::PtrsCache()) {}

    explicit MemoryManager(std::shared_ptr<UnderlyingMemPool> mempool)
        : mem_pool_(mempool),
          node_pool_(new typename UnderlyingMemPool::PtrsCache()) {}

    std::shared_ptr<UnderlyingMemPool> getUnderlyingMemPool() const {
      return mem_pool_;
    }

    // Copy (not clone) and Move are allowed but might not thread-safe
    MemoryManager(const MemoryManager& other) = default;
    MemoryManager& operator=(const MemoryManager& other) = default;
    MemoryManager& operator=(MemoryManager&& other) = default;
    MemoryManager(MemoryManager&& other) = default;

    inline size_t capacity() const {
      return mem_pool_->capacity();
    }

    inline size_t allocated() const {
      return mem_pool_->allocated();
    }

    inline NodePtr getNode(ShortPtrType short_ptr) const {
      return mem_pool_->lock(short_ptr);
    }

    inline NodePtr getNode(const NodeType* raw_node) const {
      return mem_pool_->lock(raw_node);
    }

    inline const NodeType& getConstRef(ShortPtrType short_ptr) const {
      return mem_pool_->getConstRef(short_ptr);
    }

    inline NodePtr newNode(ShortPtrType* short_ptr = nullptr) {
      auto new_node = mem_pool_->alloc();
      if (!new_node) {
        if (short_ptr) {
          *short_ptr = short_nullptr;
        }
        return nullptr;
      }
      auto new_short_ptr = new_node.shortPtr();
      if (short_ptr) {
        *short_ptr = new_short_ptr;
      }
      return std::move(new_node);
    }

    inline ShortPtrType keepNode(NodePtr&& new_node) {
      auto new_short_ptr = new_node.shortPtr();
      node_pool_->at(shortPtr2Entry(new_short_ptr))
          .become(std::move(new_node), false);
      return new_short_ptr;
    }

    inline NodePtr getKeptNode(ShortPtrType short_ptr) const {
      return node_pool_->at(shortPtr2Entry(short_ptr));
    }

    inline bool hasKeptNode(ShortPtrType short_ptr) const {
      return (node_pool_->at(shortPtr2Entry(short_ptr)) != nullptr);
    }

    inline void deleteKeptNode(ShortPtrType short_ptr) {
      ASSERT(short_ptr != short_nullptr);
      size_t entry = shortPtr2Entry(short_ptr);
      ASSERT(entry < mem_pool_->capacity());
      node_pool_->at(entry).reset();

      // if (short_ptr != short_nullptr) {
      //   size_t entry = shortPtr2Entry(short_ptr);
      //   if (entry < mem_pool_->capacity()) {
      //     node_pool_->at(entry).reset();
      //   }
      // }
    }

    void foreachAlongMemory(const std::function<void(T&)>& f) {  // NOLINT
      for (size_t i = 0;
           i < std::min(
                   node_pool_->numPages() << _mempool_n_ele_per_page_pow,
                   mem_pool_->capacity());
           ++i) {
        NodePtr ptr = node_pool_->at(i);
        if (ptr != nullptr) {
          f(ptr->data);
        }
      }
    }

   public:
    // for test
    size_t numKeptNodes() const {
      size_t num = 0;
      for (size_t i = 0;
           i < std::min(
                   node_pool_->numPages() << _mempool_n_ele_per_page_pow,
                   mem_pool_->capacity());
           ++i) {
        if (node_pool_->at(i) != nullptr) {
          ++num;
        }
      }
      return num;
    }

   private:
    std::shared_ptr<UnderlyingMemPool> mem_pool_;
    std::shared_ptr<typename UnderlyingMemPool::PtrsCache> node_pool_;
  };

  static InsertReturnType insert(
      NodePtr&& new_node, ShortPtrType new_short_ptr,
      std::atomic<MarkPtrType>* head, MemoryManager* mm,
      NodePtr* forward_fail_node = nullptr
      // `forward_fail_node` can be used to effiently recapture the
      // `new_node` when the insertion is failed.
  ) {
    // Since the node objects in the mempool might be reused, simply setting
    // init_tag to 0 is not safe (when the tag was just rolled to 0 by
    // the old reference in other thread and then reused by the current
    // thread with setting init_tag to 0).
    //
    // Note we don't have to use expensive CAS to increase tag, since the node
    // is newly allocated in the current thread, which implies no other threads
    // can write to the tag.
    //
    // The "acquire" memory_order is used for load() to ensure we can get the
    // latest updated value for tag (sync with other threads).
    ABATagType init_tag =
        new_node->mark_ptr.load(std::memory_order_acquire).tag() + 1;

    // >>> key←nodeˆ.Key;
    const T& key = new_node->data;

    FindResult r(mm);
    const auto& p = r.p;

    auto release_new_node = [&]() {
      if (forward_fail_node) {
        forward_fail_node->become(std::move(new_node), false);
      } else {
        new_node.reset(false);
      }
    };

    // >>> while true
    while (true) {
      // >>> A1: if Find(head,key) return false;
      if (find(key, head, r, mm, false)) {
        // key already exists.
        // LOGA("ListBasedSet::insert(): key already exists!");
        release_new_node();
        return InsertReturnType(false, std::move(r.cur_node));
      }

      if (r.is_head_marked) {
        release_new_node();
        return InsertReturnType(false, nullptr);
      }

      // >>> A2: nodeˆ.(Mark,Next) ← (0,cur);
      new_node->mark_ptr.store(
          MarkPtrType(0, p.next(), init_tag), std::memory_order_relaxed);

      // clang-format off
      // >>> A3: if CAS (prev, (0,cur,ptag), (0,node,ptag+1))
      // >>>        return true;
      // clang-format on
      MarkPtrType tmpp(0, p.next(), p.tag());
      if (r.prev_CAS(
              tmpp, MarkPtrType(0, new_short_ptr, p.tag() + 1),
              std::memory_order_release, std::memory_order_relaxed)) {
        auto node_to_keep = new_node;
        // if constexpr (_critical_conditions_test) {
        //   if (forward_fail_node) {
        //     std::this_thread::sleep_for(std::chrono::microseconds(1));
        //   }
        // }
        mm->keepNode(std::move(node_to_keep));
        return InsertReturnType(true, std::move(new_node));
      }
    }
  }

  template <typename KeyType>
  // Type T is required to support comparing operators (<,==) with KeyType,
  // i.e.  (T < KeyType), (T == KeyType) shuold be legal.
  static bool erase(
      const KeyType& key, std::atomic<MarkPtrType>* head, MemoryManager* mm) {
    FindResult r(mm);
    const auto& p = r.p;
    const auto& c = r.c;

    // >>> while true
    while (true) {
      // >>> B1: if !Find(head,key) return false;
      if (!find(key, head, r, mm)) {
        return false;
      }

      // Mark the node (it's going to be removed)
      //
      // clang-format off
      // >>> B2: if !CAS(&curˆ.(Mark,Next,Tag),
      // >>>             (0,next,ctag),
      // >>>             (1,next,ctag+1))
      // >>>         continue;
      // clang-format on
      MarkPtrType tmpc(0, c.next(), c.tag());
      if (!r.cur_node->mark_ptr.compare_exchange_strong(
              // set the mark bit (and increase tag).
              tmpc, MarkPtrType(1, c.next(), c.tag() + 1),
              std::memory_order_release, std::memory_order_relaxed)) {
        continue;
      }

      // if constexpr (_critical_conditions_test) {
      //   std::this_thread::sleep_for(std::chrono::microseconds(1));
      // }

      // Then try deleting the node.
      //
      // clang-format off
      // >>> B3: if CAS ( prev, (0,cur,ptag), (0,next,ptag+1))
      // >>>         DeleteNode(cur);
      // >>>     else
      // >>>         Find(head,key);
      // clang-format on
      MarkPtrType tmp(0, p.next(), p.tag());

      if (r.prev_CAS(
              // connect the prev and the next
              tmp, MarkPtrType(0, c.next(), p.tag() + 1),
              std::memory_order_release, std::memory_order_relaxed)) {
        // if constexpr (_critical_conditions_test) {
        //   std::this_thread::sleep_for(std::chrono::microseconds(1));
        // }

        mm->deleteKeptNode(p.next());
      } else {
        numInterruptedDeletes()++;  // for debug

        // If the CAS above failed (other threads modified prev), then we
        // recall find() to ensure the node be deleted (See D8: in find());
        FindResult r2(mm);
        find(key, head, r2, mm);
      }

      // >>> return true;
      return true;
    }
  }

  template <typename KeyType>
  // Type T is required to support comparing operators (<,==) with KeyType,
  // i.e.  (T < KeyType), (T == KeyType) shuold be legal.
  static NodePtr find(
      const KeyType& key, std::atomic<MarkPtrType>* head, MemoryManager* mm) {
    FindResult r(mm);
    bool found = find(key, head, r, mm);
    if (found) {
      return std::move(r.cur_node);
    } else {
      return nullptr;
    }
  }

  // foreach() will be interrupted if it detects a change in the list.
  static bool foreach (  // NOLINT
      const std::function<void(const NodePtr&)>& f,
      const std::atomic<MarkPtrType>* head, const MemoryManager* mm,
      bool check_kept_list = true) {
    const std::atomic<MarkPtrType>* const_prev = head;

    MarkPtrType p = const_prev->load(std::memory_order_relaxed);
    ABATagType ptag = p.tag();
    ShortPtrType cur = p.next();

    MarkPtrType c;
    NodePtr cur_node;
    // size_t n_marked = 0;  // for debug
    // size_t incomplete_inserts = 0;
    while (cur != short_nullptr) {
      cur_node.become(mm->getNode(cur), false);
      if (!cur_node) {
        return false;  // The list has changed, stop foreach.
      }
      c = cur_node->mark_ptr.load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_release);

      MarkPtrType tmpp(p.mark(), cur, ptag);
      // MarkPtrType tmpp(0, cur, ptag);
      if (const_prev->load(std::memory_order_relaxed) != tmpp) {
        return false;  // The list has changed, stop foreach.
      }

      if (!c.mark()) {
        // Do something with cur_node.
        // f(cur_node->data);
        if (check_kept_list && !mm->hasKeptNode(cur)) {
          // incomplete_inserts++;  // the node has not been put into the kept
          // list
        }

        f(cur_node);
      } else {
        // // cur_node is marked as deleted
        // n_marked++;
      }

      p = c;
      const_prev = &(cur_node->mark_ptr);
      ptag = p.tag();
      cur = p.next();
    }
    // LOGA(
    //     "ListBasedSet::foreach() n_marked = %d, incomplete_inserts=%d",
    //     n_marked, incomplete_inserts);
    return true;
  }

  static std::vector<NodePtr> snapshot(
      const std::atomic<MarkPtrType>* head, const MemoryManager* mm,
      bool check_kept_list = true) {
    std::vector<NodePtr> nodes;

    while (true) {
      bool unchanged = foreach (  // NOLINT
          [&](const NodePtr& node) { nodes.push_back(node); }, head, mm,
          check_kept_list);
      if (unchanged) {
        break;
      }
    }
    return nodes;
  }

 private:
  struct FindResult {
    std::atomic<MarkPtrType>* head;
    const std::atomic<MarkPtrType>* const_prev;
    MarkPtrType p;
    MarkPtrType c;
    NodePtr cur_node;
    bool is_head_marked;
    bool is_partially_inserted;  // true if the node is already in the list but
                                 // still not kept by the memory manager.

    explicit FindResult(MemoryManager* mm) : mm_(mm) {}
    bool prev_CAS(
        MarkPtrType& expected, const MarkPtrType& desired,
        std::memory_order sucess, std::memory_order failure) {
      // First we need to get mutable prev
      std::atomic<MarkPtrType>* prev;
      NodePtr tmp_prev_node;
      if (const_prev == head) {
        prev = head;
      } else {
        tmp_prev_node.become(
            mm_->getNode(NodeType::getNodeContaining(const_prev)), false);
        if (!tmp_prev_node) {
          // Other threads already removed the node.
          return false;
        }
        prev = &(tmp_prev_node->mark_ptr);
        ASSERT(prev == const_prev);
      }

      // Then do the CAS
      bool cas_ok =
          prev->compare_exchange_strong(expected, desired, sucess, failure);
      tmp_prev_node.reset(false);
      return cas_ok;
    }

   private:
    MemoryManager* mm_;
  };

  template <typename KeyType>
  // Type T is required to support comparing operators (<,==) with KeyType,
  // i.e.  (T < KeyType), (T == KeyType) shuold be legal.
  static bool find(
      const KeyType& key, std::atomic<MarkPtrType>* head, FindResult& r,
      MemoryManager* mm, bool check_kept_list = true,
      bool need_output_ptr = true) {
    const auto& p = r.p;
    const auto& c = r.c;
    r.head = head;
    r.is_partially_inserted = false;
    r.is_head_marked = false;

    // clang-format off
  try_again:  // NOLINT
    // clang-format on

    // >>> prev ← head;
    r.const_prev = head;

    // D1: p=(pmark,cur,ptag) ← *prev;
    r.p = r.const_prev->load(std::memory_order_relaxed);
    ShortPtrType cur = p.next();
    ABATagType ptag = p.tag();
    if (p.mark()) {
      r.is_head_marked = true;
      return false;  // return false directly if the head is marked.
    }

    // >>> while true
    while (true) {
      // >>> D2: if cur = null return false;
      if (UNLIKELY(cur == short_nullptr)) {
        return false;
      }
      const NodeType& cur_ref = mm->getConstRef(cur);
      // >>> D3: c=(cmark,next,ctag) ← cur.^(Mark,Next,Tag);
      r.c = cur_ref.mark_ptr.load(std::memory_order_relaxed);
      // >>> D4: ckey←cur.^Key;
      const T& ckey = cur_ref.data;
      bool ckey_lt_key = (ckey < key);
      bool ckey_eq_key = (ckey == key);

      // To ensure reads/writes during D2 to D4 not reordered after D5.
      std::atomic_thread_fence(std::memory_order_release);

      // >>> D5: if *prev != (0,cur,ptag) goto try again;
      // check whether prev has changed
      MarkPtrType tmpp(0, cur, ptag);
      if (UNLIKELY(r.const_prev->load(std::memory_order_relaxed) != tmpp)) {
        goto try_again;  // NOLINT
      }

      // >>> if !cmark
      if (LIKELY(!c.mark())) {
        // If the current node is valid (not going to be removed).

        // >>> D6: if ckey≥key return ckey = key;
        if (UNLIKELY(!ckey_lt_key)) {
          // return ckey == key;
          if (ckey_eq_key) {
            if (LIKELY(need_output_ptr)) {
              // Increase cur's ref_count (by calling getNode())
              // only when it's needed.
              r.cur_node.become(mm->getNode(cur), false);
              std::atomic_thread_fence(std::memory_order_release);

              // check whether prev has changed
              if (UNLIKELY(
                      r.const_prev->load(std::memory_order_relaxed) != tmpp)) {
                r.cur_node.reset(false);
                goto try_again;  // NOLINT
              }
            }
            if (check_kept_list && !mm->hasKeptNode(cur)) {
              r.is_partially_inserted = true;
              return false;  // the node has not been put into the kept list
            }
            return true;
          } else {
            return false;
          }
        }

        // >>> D7: prev ← &curˆ.(Mark,Next,Tag);
        r.const_prev = &(cur_ref.mark_ptr);
        // Since we didn't increase the ref_count of cur (we just got
        // its ref by getConstRef), so the cur node may have been
        // deleted (mark->1) or even the same address heen reused, i.e.
        // reallocated and modified (tag changed).
        // It seems a little dangerous, however it can also be handled
        // in the next loop at "D5:".
      } else {
        // If the current node is outdated (going to be removed), try
        // removing the node.

        // clang-format off
        // >>> D8: if CAS ( prev, (0,cur,ptag) , (0,next,ptag+1))
        // >>>        {DeleteNode(cur); ctag ←ptag+1;}
        // >>>     else
        // >>>        goto try again;
        // clang-format on
        if (LIKELY(r.prev_CAS(
                tmpp, MarkPtrType(0, c.next(), ptag + 1),
                std::memory_order_release, std::memory_order_relaxed))) {
          // DeleteNode(cur);
          mm->deleteKeptNode(cur);

          numFindHelpedDeletes()++;  // for debug

          // ctag ← ptag+1;
          r.c.setTag(ptag + 1);
        } else {
          goto try_again;  // NOLINT
        }
      }
      // >>> D9: (pmark,cur,ptag) ← (cmark,next,ctag);
      r.p = c;  // Note "r.p=c" ensures r.p and *prev (atomic) have the same
                // next() & tag(), but their mark()'s might differ. However
                // this doesn't matter since p.mark() won't be used anymore.

      // set cur cand ptag
      cur = p.next();
      ptag = p.tag();
    }
  }

 public:
  static std::atomic<size_t>& numInterruptedDeletes() {
    static std::atomic<size_t> numInterruptedDeletes(0);  // NOLINT
    return numInterruptedDeletes;
  }

  static std::atomic<size_t>& numFindHelpedDeletes() {
    static std::atomic<size_t> numFindHelpedDeletes(0);  // NOLINT
    return numFindHelpedDeletes;
  }

 private:
  mutable std::atomic<MarkPtrType> head_;
  mutable MemoryManager mm_;
};

///// unit tests ////
template <
    typename T, size_t _mempool_n_ele_per_page_pow = 8,
    size_t _mempool_short_ptr_bits = 24, size_t _mempool_aba_tag_bits = 16>
using ListBasedSetForUnitTest = ListBasedSet<
    T, _mempool_n_ele_per_page_pow, _mempool_short_ptr_bits,
    _mempool_aba_tag_bits, true>;

}  // namespace sk4slam
