#pragma once

#include <Eigen/Core>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

// Usually, LieGroup::data() should returns a raw pointer of type `Scalar*` or
// `const Scalar*`, as the code of the class Rp shows (See Rp::data()).
// However, sometimes the data might not be stored in a continuous memory
// block, but been split into several parts, especially when we use product
// Lie Groups, e.g. a SE(3) element can be composed of a SO(3) element (a
// rotation) and a translation vector. In this case, we can use
// LieGroupDataIterator as the return type of LieGroup::data(), which allows
// us to store the data in several non-continuous memory blocks.
//
template <typename ScalarType>
struct LieGroupDataIterator {
  using Scalar = ScalarType;
  using DataPart = std::pair<Scalar*, size_t>;

  // NOTE: The internal ref_count of the data_parts is not thread-safe for
  //       performance considerations, and that's why we did not use
  //       `std::shared_ptr` here. (`shared_ptr` needs atomic operations
  //       to increment/decrement the ref-count, which is expensive).
  //
  //       It is the caller's responsibility to make sure the iterator,
  //       i.e. the return value of LieGroup::data(), is not used in a
  //       multi-threaded environment.
  explicit LieGroupDataIterator(std::vector<DataPart> data_parts)
      : shared_data_parts_(new SharedDataParts(std::move(data_parts))) {}

  Scalar& operator*() const {
    // ASSERT(current_part_ < parts().size());
    return parts()[current_part_].first[current_index_];
  }

  LieGroupDataIterator& operator++() {
    ++current_index_;
    if (current_index_ == parts()[current_part_].second) {
      ++current_part_;
      current_index_ = 0;
    }
    return *this;
  }

  LieGroupDataIterator operator++(int) {
    LieGroupDataIterator tmp(*this);
    ++(*this);
    return tmp;
  }

  template <typename... _LieGroups>
  static LieGroupDataIterator ConcatenateDataForLieGroups(_LieGroups&... gs) {
    LieGroupDataIterator data;
    (data << ... << gs);
    return data;
  }

  LieGroupDataIterator(const LieGroupDataIterator& other)
      : shared_data_parts_(other.shared_data_parts_),
        current_part_(other.current_part_),
        current_index_(other.current_index_) {
    shared_data_parts_->incRef();
  }

  LieGroupDataIterator& operator=(const LieGroupDataIterator& other) {
    if (this != &other) {
      release();
      shared_data_parts_ = other.shared_data_parts_;
      current_part_ = other.current_part_;
      current_index_ = other.current_index_;
      shared_data_parts_->incRef();
    }
    return *this;
  }

  ~LieGroupDataIterator() {
    release();
  }

  // It's not recommended to use this function to traverse the data, since
  // it can be slow if the data is split into several parts. Use ++iterator
  // instead.
  Scalar& operator[](int i) const {
    LOGA("LieGroupDataIterator::openator[%d]", i);
    auto& parts = this->parts();
    int part_idx = current_part_;
    int target_idx = current_index_ + i;

    while (target_idx >= parts[part_idx].second) {
      target_idx -= parts[part_idx].second;
      ++part_idx;
      ASSERT(part_idx < parts.size());
    }
    return parts[part_idx].first[target_idx];
  }

 private:
  LieGroupDataIterator() : shared_data_parts_(new SharedDataParts()) {}

  template <typename _LieGroup>
  LieGroupDataIterator& operator<<(_LieGroup& g) {
    auto p = g.data();

    // If p itself is a raw pointer (Scalar* or const Scalar*), then we can
    // directly add it to this->shared_data_parts_;
    // Otherwise, p is of type LieGroupDataIterator and it contains several
    // data parts, we add all the data parts of p to this->shared_data_parts_.
    using p_type = decltype(p);
    if constexpr (std::is_pointer<p_type>::value) {
      // p is a raw pointer
      parts().emplace_back(DataPart(p, g.kAmbientDim));
    } else {
      // p is of type LieGroupDataIterator
      auto& parts = this->parts();
      parts.insert(parts.end(), p.parts().begin(), p.parts().end());
    }
    return *this;
  }

 private:
  struct SharedDataParts {
    SharedDataParts() : ref_count_(1) {}
    explicit SharedDataParts(std::vector<DataPart>&& data_parts)
        : ref_count_(1), data_parts_(std::move(data_parts)) {}
    int incRef() {
      return ref_count_++;
    }
    int decRef() {
      return ref_count_--;
    }
    int ref_count_;
    std::vector<DataPart> data_parts_;
  };

  std::vector<DataPart>& parts() const {
    return shared_data_parts_->data_parts_;
  }

  void release() {
    int n_ref_before_release = shared_data_parts_->decRef();
    if (1 >= n_ref_before_release) {
      ASSERT(1 == n_ref_before_release);
      LOGA(
          "LieGroupDataIterator::release(): n_ref_before_release=1, delete "
          "shared_data_parts_");
      delete shared_data_parts_;
    }
  }

 private:
  SharedDataParts* shared_data_parts_;
  int current_part_ = 0;
  int current_index_ = 0;
};

}  // namespace sk4slam
