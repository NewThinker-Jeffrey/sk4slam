////////////////////////////////////////////////////////////////////////////////
/// @file    deque.h
/// @brief A custom implementation of a double-ended queue (deque) that mimics
/// the interface of `std::deque`.
///
/// The class `sk4slam::Deque` supports user-defined block sizes (number of
/// elements per block), allowing more control over memory allocation. It
/// provides efficient insertion and deletion at both ends, random access, and
/// the ability to dynamically expand.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "sk4slam_basic/logging.h"

namespace sk4slam {

namespace deque_internal {
/// @brief  Const iterator class for Deque.
template <typename Deque>
class const_iterator {
  using T = typename Deque::value_type;

 public:
  /// @brief Constructs a const iterator pointing to a specific index in the
  /// deque.
  const_iterator(const Deque* deque, std::ptrdiff_t pos)
      : deque(deque), pos(pos) {}

  /// @brief Dereferences the iterator.
  const T& operator*() const {
    return (*deque)[pos];
  }

  /// @brief Pre-increments the iterator.
  const_iterator& operator++() {
    ++pos;
    return *this;
  }

  /// @brief Post-increments the iterator.
  const_iterator operator++(int) {
    const_iterator temp = *this;
    ++(*this);
    return temp;
  }

  /// @brief Pre-decrements the iterator.
  const_iterator& operator--() {
    --pos;
    return *this;
  }

  /// @brief Post-decrements the iterator.
  const_iterator operator--(int) {
    const_iterator temp = *this;
    --(*this);
    return temp;
  }

  /// @brief Adds an offset to the iterator.
  const_iterator operator+(std::ptrdiff_t offset) const {
    return const_iterator(deque, pos + offset);
  }

  /// @brief Subtracts an offset from the iterator.
  const_iterator operator-(std::ptrdiff_t offset) const {
    return const_iterator(deque, pos - offset);
  }

  /// @brief Computes the distance between two iterators.
  std::ptrdiff_t operator-(const const_iterator& other) const {
    return pos - other.pos;
  }

  /// @brief Equality comparison.
  bool operator==(const const_iterator& other) const {
    return deque == other.deque && pos == other.pos;
  }

  /// @brief Inequality comparison.
  bool operator!=(const const_iterator& other) const {
    return !(*this == other);
  }

 private:
  const Deque* deque;  ///< Pointer to the associated deque.
  std::ptrdiff_t pos;  ///< Logical position in the deque.
};

/// @brief Iterator class for Deque.
template <typename Deque>
class iterator {
  using T = typename Deque::value_type;
  using _const_iterator = deque_internal::const_iterator<Deque>;

 public:
  /// @brief Constructs an iterator pointing to a specific index in the deque.
  iterator(Deque* deque, std::ptrdiff_t pos) : deque(deque), pos(pos) {}

  /// @brief Dereferences the iterator.
  T& operator*() {
    return (*deque)[pos];
  }

  /// @brief Pre-increments the iterator.
  iterator& operator++() {
    ++pos;
    return *this;
  }

  /// @brief Post-increments the iterator.
  iterator operator++(int) {
    iterator temp = *this;
    ++(*this);
    return temp;
  }

  /// @brief Pre-decrements the iterator.
  iterator& operator--() {
    --pos;
    return *this;
  }

  /// @brief Post-decrements the iterator.
  iterator operator--(int) {
    iterator temp = *this;
    --(*this);
    return temp;
  }

  /// @brief Adds an offset to the iterator.
  iterator operator+(std::ptrdiff_t offset) const {
    return iterator(deque, pos + offset);
  }

  /// @brief Subtracts an offset from the iterator.
  iterator operator-(std::ptrdiff_t offset) const {
    return iterator(deque, pos - offset);
  }

  /// @brief Computes the distance between two iterators.
  std::ptrdiff_t operator-(const iterator& other) const {
    return pos - other.pos;
  }

  /// @brief Equality comparison.
  bool operator==(const iterator& other) const {
    return deque == other.deque && pos == other.pos;
  }

  /// @brief Inequality comparison.
  bool operator!=(const iterator& other) const {
    return !(*this == other);
  }

  /// @brief Implicit conversion to const_iterator.
  operator _const_iterator() const {
    return _const_iterator(
        deque,
        pos);  // Create a const_iterator with the same deque and position
  }

 private:
  Deque* deque;        ///< Pointer to the associated deque.
  std::ptrdiff_t pos;  ///< Logical position in the deque.
};

/// @brief Const reverse iterator class for Deque.
template <typename Deque>
class const_reverse_iterator {
  using T = typename Deque::value_type;

 public:
  /// @brief Constructs a const reverse iterator pointing to a specific index in
  /// the deque.
  const_reverse_iterator(const Deque* deque, std::ptrdiff_t pos)
      : deque(deque), pos(pos) {}

  /// @brief Dereferences the iterator.
  const T& operator*() const {
    return (*deque)[pos];
  }

  /// @brief Pre-increments the reverse iterator.
  const_reverse_iterator& operator++() {
    --pos;
    return *this;
  }

  /// @brief Post-increments the reverse iterator.
  const_reverse_iterator operator++(int) {
    const_reverse_iterator temp = *this;
    --(*this);
    return temp;
  }

  /// @brief Pre-decrements the reverse iterator.
  const_reverse_iterator& operator--() {
    ++pos;
    return *this;
  }

  /// @brief Post-decrements the reverse iterator.
  const_reverse_iterator operator--(int) {
    const_reverse_iterator temp = *this;
    ++(*this);
    return temp;
  }

  /// @brief Adds an offset to the reverse iterator.
  const_reverse_iterator operator+(std::ptrdiff_t offset) const {
    return const_reverse_iterator(deque, pos - offset);
  }

  /// @brief Subtracts an offset from the reverse iterator.
  const_reverse_iterator operator-(std::ptrdiff_t offset) const {
    return const_reverse_iterator(deque, pos + offset);
  }

  /// @brief Computes the distance between two reverse iterators.
  std::ptrdiff_t operator-(const const_reverse_iterator& other) const {
    return other.pos - pos;
  }

  /// @brief Equality comparison.
  bool operator==(const const_reverse_iterator& other) const {
    return deque == other.deque && pos == other.pos;
  }

  /// @brief Inequality comparison.
  bool operator!=(const const_reverse_iterator& other) const {
    return !(*this == other);
  }

 private:
  const Deque* deque;  ///< Pointer to the associated deque.
  std::ptrdiff_t pos;  ///< Logical position in the deque.
};

/// @brief Reverse iterator class for Deque.
template <typename Deque>
class reverse_iterator {
  using T = typename Deque::value_type;
  using _const_reverse_iterator = deque_internal::const_reverse_iterator<Deque>;

 public:
  /// @brief Constructs a reverse iterator pointing to a specific index in the
  /// deque.
  reverse_iterator(Deque* deque, std::ptrdiff_t pos) : deque(deque), pos(pos) {}

  /// @brief Dereferences the iterator.
  T& operator*() {
    return (*deque)[pos];
  }

  /// @brief Pre-increments the reverse iterator.
  reverse_iterator& operator++() {
    --pos;
    return *this;
  }

  /// @brief Post-increments the reverse iterator.
  reverse_iterator operator++(int) {
    reverse_iterator temp = *this;
    --(*this);
    return temp;
  }

  /// @brief Pre-decrements the reverse iterator.
  reverse_iterator& operator--() {
    ++pos;
    return *this;
  }

  /// @brief Post-decrements the reverse iterator.
  reverse_iterator operator--(int) {
    reverse_iterator temp = *this;
    ++(*this);
    return temp;
  }

  /// @brief Adds an offset to the reverse iterator.
  reverse_iterator operator+(std::ptrdiff_t offset) const {
    return reverse_iterator(deque, pos - offset);
  }

  /// @brief Subtracts an offset from the reverse iterator.
  reverse_iterator operator-(std::ptrdiff_t offset) const {
    return reverse_iterator(deque, pos + offset);
  }

  /// @brief Computes the distance between two reverse iterators.
  std::ptrdiff_t operator-(const reverse_iterator& other) const {
    return other.pos - pos;
  }

  /// @brief Equality comparison.
  bool operator==(const reverse_iterator& other) const {
    return deque == other.deque && pos == other.pos;
  }

  /// @brief Inequality comparison.
  bool operator!=(const reverse_iterator& other) const {
    return !(*this == other);
  }

  /// @brief Implicit conversion to const_reverse_iterator.
  operator _const_reverse_iterator() const {
    return _const_reverse_iterator(deque, pos);
  }

 private:
  Deque* deque;        ///< Pointer to the associated deque.
  std::ptrdiff_t pos;  ///< Logical position in the deque.
};

}  // namespace deque_internal

/// @brief A custom implementation of a double-ended queue (deque) that mimics
/// the interface of `std::deque`.
///
/// This class supports user-defined block sizes (number of elements per block),
/// allowing more control over memory allocation. It provides efficient
/// insertion and deletion at both ends, random access, and the ability to
/// dynamically expand.
///
/// @tparam T The type of elements stored in the deque.
/// @tparam _block_size The number of elements stored in each block. This must
/// be greater than 0.
template <typename T, std::size_t _block_size = 64>
class Deque {
 public:
  using value_type = T;
  using iterator = deque_internal::iterator<Deque>;
  using const_iterator = deque_internal::const_iterator<Deque>;
  using reverse_iterator = deque_internal::reverse_iterator<Deque>;
  using const_reverse_iterator = deque_internal::const_reverse_iterator<Deque>;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;

  static constexpr std::size_t kBlockSize = _block_size;
  static constexpr std::size_t kBlockSizeMask = kBlockSize - 1;
  // static constexpr std::size_t kBlockSizeShift =

  /// @brief Returns an iterator pointing to the first element of the deque.
  iterator begin() {
    return iterator(this, 0);  // Start from the first logical element
  }

  /// @brief Returns an iterator pointing to the past-the-end element of the
  /// deque.
  iterator end() {
    return iterator(this, size());  // Past-the-end element
  }

  /// @brief Returns a const_iterator pointing to the first element of the
  /// deque.
  const_iterator begin() const {
    return const_iterator(this, 0);  // Start from the first logical element
  }

  /// @brief Returns a const_iterator pointing to the past-the-end element of
  /// the deque.
  const_iterator end() const {
    return const_iterator(this, size());  // Past-the-end element
  }

  /// @brief Returns a reverse_iterator pointing to the last element of the
  /// deque.
  reverse_iterator rbegin() {
    return reverse_iterator(
        this, size() - 1);  // Start from the last logical element
  }

  /// @brief Returns a reverse_iterator pointing to the past-the-end element of
  /// the deque.
  reverse_iterator rend() {
    return reverse_iterator(
        this, 0ul - 1);  // Past-the-end element (before the first element)
  }

  /// @brief Returns a const_reverse_iterator pointing to the last element of
  /// the deque.
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(
        this, size() - 1);  // Start from the last logical element
  }

  /// @brief Returns a const_reverse_iterator pointing to the past-the-end
  /// element of the deque.
  const_reverse_iterator rend() const {
    return const_reverse_iterator(
        this, 0ul - 1);  // Past-the-end element (before the first element)
  }

  /// @brief Default constructor
  Deque() : front_index(0), back_index(0) {}

  /// @brief Copy constructor
  Deque(const Deque& other) : front_index(0), back_index(other.size()) {
    size_t n_blocks = (other.size() + kBlockSize - 1) / kBlockSize;
    blocks.reserve(n_blocks);
    for (std::size_t i = 0; i < n_blocks; ++i) {
      blocks.push_back(allocate_block());
    }
    for (std::size_t i = 0; i < other.size(); ++i) {
      new (get_element_ptr(i)) T(other[i]);
    }
  }

  /// @brief Copy assignment operator
  Deque& operator=(const Deque& other) {
    if (this == &other)
      return *this;
    clear();
    size_t n_blocks = (other.size() + kBlockSize - 1) / kBlockSize;
    blocks.reserve(n_blocks);
    front_index = 0;
    back_index = other.size();
    for (std::size_t i = 0; i < n_blocks; ++i) {
      blocks.push_back(allocate_block());
    }
    for (std::size_t i = 0; i < other.size(); ++i) {
      new (get_element_ptr(i)) T(other[i]);
    }
    return *this;
  }

  /// @brief Move constructor
  Deque(Deque&& other) noexcept
      : front_index(other.front_index),
        back_index(other.back_index),
        blocks(std::move(other.blocks)) {
    other.blocks.clear();
    other.front_index = 0;
    other.back_index = 0;
  }

  /// @brief Move assignment operator
  Deque& operator=(Deque&& other) noexcept {
    if (this == &other)
      return *this;
    clear();
    front_index = other.front_index;
    back_index = other.back_index;
    blocks = std::move(other.blocks);
    other.blocks.clear();
    other.front_index = 0;
    other.back_index = 0;
    return *this;
  }

  /// @brief Destructor
  ~Deque() {
    clear();
  }

  /// @brief Push an element to the back
  void push_back(const T& value) {
    emplace_back(value);
  }

  /// @brief Push an element to the front
  void push_front(const T& value) {
    emplace_front(value);
  }

  /// @brief Emplace an element at the back
  template <typename... Args>
  void emplace_back(Args&&... args) {
    ensure_back_capacity();
    new (get_element_ptr(back_index)) T(std::forward<Args>(args)...);
    ++back_index;
  }

  /// @brief Emplace an element at the front
  template <typename... Args>
  void emplace_front(Args&&... args) {
    ensure_front_capacity();
    --front_index;
    new (get_element_ptr(front_index)) T(std::forward<Args>(args)...);
  }

  /// @brief Pop an element from the back
  void pop_back() {
    if (size() == 0) {
      throw std::out_of_range("Deque is empty");
    }
    --back_index;                           // Move back index one step back
    T* elem = get_element_ptr(back_index);  // Get pointer to the last element
    elem->~T();                             // Call destructor of the element
  }

  /// @brief Pop an element from the front
  void pop_front() {
    if (size() == 0) {
      throw std::out_of_range("Deque is empty");
    }
    T* elem = get_element_ptr(front_index);  // Get pointer to the first element
    elem->~T();                              // Call destructor of the element
    ++front_index;  // Move front index one step forward
  }

  /// @brief Inserts an element at the specified position.
  /// @param pos A const_iterator pointing to the position where the element
  /// will be inserted.
  /// @param value The value to insert.
  /// @return An iterator pointing to the inserted element.
  template <typename T_or_Tref>
  iterator insert(const_iterator pos, T_or_Tref&& value) {
    std::size_t index = pos - begin();  // Calculate the logical index

    // Special case: Insert at the front
    if (index == 0) {
      push_front(std::forward<T_or_Tref>(value));
      return begin();
    }

    // Special case: Insert at the back
    if (index == size()) {
      push_back(std::forward<T_or_Tref>(value));
      return iterator(this, size() - 1);
    }

    // Choose the optimal direction to move data
    if (index < size() / 2) {
      // Move elements left (toward front_index)
      ensure_front_capacity();
      --front_index;
      for (std::size_t i = 0; i < index; ++i) {
        T* dest = get_element_ptr(front_index + i);
        T* src = get_element_ptr(front_index + i + 1);
        new (dest) T(std::move(*src));
        src->~T();
      }
    } else {
      // Move elements right (toward back_index)
      ensure_back_capacity();
      for (std::size_t i = size(); i > index; --i) {
        T* dest = get_element_ptr(front_index + i);
        T* src = get_element_ptr(front_index + i - 1);
        new (dest) T(std::move(*src));
        src->~T();
      }
      ++back_index;
    }

    // Insert the new element
    T* insert_pos = get_element_ptr(front_index + index);
    new (insert_pos) T(std::forward<T_or_Tref>(value));

    return iterator(this, index);
  }

  /// @brief Inserts `count` copies of `value` at the specified position.
  /// @param pos A const_iterator pointing to the position where the elements
  /// will be inserted.
  /// @param count The number of copies of `value` to insert.
  /// @param value The value to insert.
  /// @return An iterator pointing to the first inserted element.
  iterator insert(const_iterator pos, size_t count, const T& value) {
    if (count == 0) {
      return iterator(this, pos - begin());  // No insertion
    }
    if (count == 1) {
      return insert(pos, value);  // Single element insertion
    }

    std::size_t index = pos - begin();
    // Choose the optimal direction to move data
    if (index < size() / 2) {
      // Move elements left (toward front_index)
      ensure_front_capacity(count);
      front_index -= count;
      for (std::size_t i = 0; i < index; ++i) {
        T* dest = get_element_ptr(front_index + i);
        T* src = get_element_ptr(front_index + i + count);
        new (dest) T(std::move(*src));
        src->~T();
      }
    } else {
      // Move elements right (toward back_index)
      ensure_back_capacity(count);
      for (std::size_t i = size(); i > index; --i) {
        T* dest = get_element_ptr(front_index + i + count - 1);
        T* src = get_element_ptr(front_index + i - 1);
        new (dest) T(std::move(*src));
        src->~T();
      }
      back_index += count;
    }

    // Insert the new elements
    for (size_t i = 0; i < count; ++i) {
      T* insert_pos = get_element_ptr(front_index + index + i);
      new (insert_pos) T(value);
    }

    return iterator(this, index);
  }

  /// @brief Inserts a range of elements at the specified position.
  /// @tparam InputIt The type of the input iterator.
  /// @param pos A const_iterator pointing to the position where the range will
  /// be inserted.
  /// @param first An iterator to the beginning of the range.
  /// @param last An iterator to the end of the range.
  /// @return An iterator pointing to the first inserted element.
  template <class InputIt>
  iterator insert(const_iterator pos, InputIt first, InputIt last) {
    std::size_t index = pos - begin();  // Logical index of insertion
    std::size_t count = std::distance(first, last);

    if (count == 0) {
      return iterator(this, index);  // No insertion
    }

    if (count == 1) {
      return insert(pos, *first);  // Single element insertion
    }

    // Choose the optimal direction to move data
    if (index < size() / 2) {
      // Move elements left (toward front_index)
      ensure_front_capacity(count);
      front_index -= count;
      for (std::size_t i = 0; i < index; ++i) {
        T* dest = get_element_ptr(front_index + i);
        T* src = get_element_ptr(front_index + i + count);
        new (dest) T(std::move(*src));
        src->~T();
      }
    } else {
      // Move elements right (toward back_index)
      ensure_back_capacity(count);
      for (std::size_t i = size(); i > index; --i) {
        T* dest = get_element_ptr(front_index + i + count - 1);
        T* src = get_element_ptr(front_index + i - 1);
        new (dest) T(std::move(*src));
        src->~T();
      }
      back_index += count;
    }

    // Insert the new elements
    size_t i = 0;
    for (auto it = first; it != last; ++it, ++i) {
      T* insert_pos = get_element_ptr(front_index + index + i);
      new (insert_pos) T(*it);
    }

    return iterator(this, index);
  }

  /// @brief Inserts elements from an initializer list at the specified
  /// position.
  /// @param pos A const_iterator pointing to the position where the elements
  /// will be inserted.
  /// @param ilist An initializer list containing the elements to insert.
  /// @return An iterator pointing to the first inserted element.
  iterator insert(const_iterator pos, std::initializer_list<T> ilist) {
    return insert(
        pos, ilist.begin(), ilist.end());  // Forward to range-based insert
  }

  /// @brief Removes the element at the specified position.
  /// @param pos A const_iterator pointing to the element to be removed.
  /// @return An iterator pointing to the element following the erased element.
  iterator erase(const_iterator pos) {
    // First handle the special cases of removing the front or back element
    if (pos == begin()) {
      pop_front();
      return begin();  // Return iterator to the new front element
    } else if (pos == end() - 1) {
      pop_back();
      return end();  // Return iterator to the end of the deque.
    }

    std::size_t index = pos - begin();  // Get the logical index of the element

    if (index >= size()) {
      throw std::out_of_range("Deque::erase() - iterator out of range");
    }

    // Move elements to fill the gap
    for (std::size_t i = front_index + index; i < back_index - 1; ++i) {
      T* current = get_element_ptr(i);
      T* next = get_element_ptr(i + 1);
      current->~T();  // Destroy the current element
      new (current) T(
          std::move(*next));  // Move the next element into the current position
    }

    // Destroy the last element
    get_element_ptr(back_index - 1)->~T();

    --back_index;

    return iterator(this, index);  // Return iterator to the next element
  }

  /// @brief Removes a range of elements from the deque.
  /// @param first A const_iterator pointing to the first element to remove.
  /// @param last A const_iterator pointing to the element after the last
  /// element to remove.
  /// @return An iterator pointing to the element following the last erased
  /// element.
  iterator erase(const_iterator first, const_iterator last) {
    // First handle the special cases of removing from the front or back
    if (first == begin()) {
      std::size_t num_elements_to_remove = last - first;
      while (num_elements_to_remove--) {
        pop_front();
      }
      return begin();  // Return iterator to the new front element
    } else if (last == end()) {
      std::size_t num_elements_to_remove = last - first;
      while (num_elements_to_remove--) {
        pop_back();
      }
      return end();  // Return iterator to the end of the deque.
    }

    std::size_t start_index = first - begin();
    std::size_t end_index = last - begin();

    if (start_index > end_index || end_index > size()) {
      throw std::out_of_range("Deque::erase() - range out of bounds");
    }

    std::size_t num_elements_to_remove = end_index - start_index;

    // Move elements to fill the gap
    for (std::size_t i = front_index + start_index;
         i < back_index - num_elements_to_remove; ++i) {
      T* current = get_element_ptr(i);
      T* next = get_element_ptr(i + num_elements_to_remove);
      current->~T();  // Destroy the current element
      new (current) T(
          std::move(*next));  // Move the next element into the current position
    }

    // Destroy the remaining elements
    for (std::size_t i = back_index - num_elements_to_remove; i < back_index;
         ++i) {
      get_element_ptr(i)->~T();
    }

    back_index -= num_elements_to_remove;

    return iterator(
        this,
        start_index);  // Return iterator to the element after the erased range
  }

  /// @brief Shrinks the deque to fit its size, releasing unused memory.
  void shrink_to_fit() {
    // Calculate the number of elements
    std::size_t size = back_index - front_index;

    // Calculate the number of blocks needed
    std::size_t needed_blocks = (size + kBlockSize - 1) / kBlockSize;

    // If the current blocks array is already minimal, no action needed
    if (needed_blocks == blocks.size()) {
      return;
    }

    // Create a new block array and allocate only the needed blocks
    std::vector<Block*> new_blocks(needed_blocks, nullptr);
    for (std::size_t i = 0; i < needed_blocks; ++i) {
      new_blocks[i] = allocate_block();
    }

    // Copy data to the new block array
    for (std::size_t i = 0; i < size; ++i) {
      T* src = get_element_ptr(front_index + i);
      T* dest =
          reinterpret_cast<T*>(&new_blocks[i / kBlockSize][i % kBlockSize]);
      new (dest) T(std::move(*src));  // Move construct the element
      src->~T();                      // Destroy the old element
    }

    // Release old blocks
    for (auto& block : blocks) {
      if (block) {
        free_block(block);
        block = nullptr;
      }
    }

    // Replace blocks with the new array and update indices
    blocks = std::move(new_blocks);
    front_index = 0;
    back_index = size;
  }

  /// @brief Trims unused blocks and shrinks blocks' capacity if necessary.
  ///
  /// This method reduces memory usage by releasing unused blocks at the front
  /// and back of the deque if they exceed the specified limits. It also shrinks
  /// the internal `blocks` vector capacity if it is significantly larger than
  /// its size.
  ///
  /// @note This method is an extension beyond the standard `std::deque`
  /// implementation, which does not provide fine-grained control over memory
  /// management at the block level.
  ///
  /// @param max_front_blocks Maximum number of unused blocks allowed at the
  /// front.
  /// @param max_back_blocks Maximum number of unused blocks allowed at the
  /// back.
  void trim_unused_blocks(int max_front_blocks, int max_back_blocks) {
    // Calculate unused blocks at the front
    size_t unused_front_blocks = front_index / kBlockSize;

    // Calculate unused blocks at the back
    size_t unused_back_blocks =
        (blocks.size() * kBlockSize - back_index) / kBlockSize;

    // Release unused back blocks if they exceed the limit
    if (unused_back_blocks > max_back_blocks) {
      size_t blocks_to_remove = unused_back_blocks - max_back_blocks;
      // LOGA(
      //     "Trimming %zu blocks from the back, unused_back_blocks = %zu, "
      //     "max_back_blocks = %d",
      //     blocks_to_remove, unused_back_blocks, max_back_blocks);

      // Delete the unused blocks
      for (size_t i = 0; i < blocks_to_remove; ++i) {
        free_block(blocks[blocks.size() - 1 - i]);
        blocks[blocks.size() - 1 - i] = nullptr;
      }

      // Remove the blocks from the vector (Erasing from the end won't cause
      // reallocation)
      blocks.erase(blocks.end() - blocks_to_remove, blocks.end());
    }

    // Release unused front blocks if they exceed the limit
    if (unused_front_blocks > max_front_blocks) {
      size_t blocks_to_remove = unused_front_blocks - max_front_blocks;
      // LOGA(
      //     "Trimming %zu blocks from the front, unused_front_blocks = %zu, "
      //     "max_front_blocks = %d",
      //     blocks_to_remove, unused_front_blocks, max_front_blocks);

      // Delete the unused blocks
      for (size_t i = 0; i < blocks_to_remove; ++i) {
        free_block(blocks[i]);
        blocks[i] = nullptr;
      }

      // Remove the blocks from the vector
      std::vector<Block*> new_blocks(
          blocks.begin() + blocks_to_remove, blocks.end());
      blocks.swap(new_blocks);  // Swap the contents to reduce capacity

      // Adjust front_index and back_index
      front_index -= blocks_to_remove * kBlockSize;
      back_index -= blocks_to_remove * kBlockSize;
    }

    // Check if blocks' capacity is much larger than its size
    if (blocks.capacity() > 2 * blocks.size()) {
      // Shrink to fit: Create a new vector with the same elements but smaller
      // capacity
      std::vector<Block*> new_blocks(blocks.begin(), blocks.end());
      blocks.swap(new_blocks);  // Swap the contents to reduce capacity
    }
  }

  /// @brief Trims unused blocks to an optimal state based on current size.
  ///
  /// Retains at most `max(N/4, 1)` blocks at the front and back, where `N` is
  /// the minimum number of blocks required to hold the current size.
  /// If `front_index` is in the latter half of its block, that block is
  /// included in the retained range. Similarly, if `back_index` is in the first
  /// half of its block, that block is included in the retained range.
  void trim_to_optimal() {
    // Calculate the minimum number of blocks needed to hold current size
    size_t num_elements = back_index - front_index;  // Total number of elements
    size_t min_blocks =
        (num_elements + kBlockSize - 1) / kBlockSize;  // Minimum blocks needed

    // Calculate the base retain count
    size_t retain_count = std::max(min_blocks / 4, size_t(1));
    // LOGA(
    //     "retain_count = %d for num_elements = %d", retain_count,
    //     num_elements);

    // Determine if the front block should be retained
    bool front_retain_extra = (front_index % kBlockSize) >= kBlockSize / 2;
    // LOGA(
    //     "front_retain_extra = %d for front_index = %d", front_retain_extra,
    //     front_index);

    // Determine if the back block should be retained
    bool back_retain_extra = (back_index % kBlockSize) < kBlockSize / 2;
    // LOGA(
    //     "back_retain_extra = %d for back_index = %d", back_retain_extra,
    //     back_index);

    // Adjust retain counts based on extra retention rules
    size_t front_retain_count = retain_count - (front_retain_extra ? 1 : 0);
    size_t back_retain_count = retain_count - (back_retain_extra ? 1 : 0);
    // LOGA(
    //     "front_retain_count = %d, back_retain_count = %d",
    //     front_retain_count, back_retain_count);

    // Call trim_unused_blocks() to handle the actual trimming
    trim_unused_blocks(front_retain_count, back_retain_count);
  }

  /// @brief Get size of the deque
  std::size_t size() const {
    return back_index - front_index;
  }

  /// @brief Check if the deque is empty
  bool empty() const {
    return size() == 0;
  }

  /// @brief Access element by index
  T& operator[](std::size_t index) {
    if (index >= size())
      throw std::out_of_range("Index out of range");
    return *get_element_ptr(front_index + index);
  }

  const T& operator[](std::size_t index) const {
    if (index >= size())
      throw std::out_of_range("Index out of range");
    return *get_element_ptr(front_index + index);
  }

  T& at(std::size_t index) {
    if (index >= size())
      throw std::out_of_range("Index out of range");
    return *get_element_ptr(front_index + index);
  }

  const T& at(std::size_t index) const {
    if (index >= size())
      throw std::out_of_range("Index out of range");
    return *get_element_ptr(front_index + index);
  }

  /// @brief Returns a reference to the first element in the deque.
  T& front() {
    if (size() == 0) {
      throw std::out_of_range("Deque::front() - deque is empty");
    }
    return *get_element_ptr(front_index);
  }

  /// @brief Returns a const reference to the first element in the deque.
  const T& front() const {
    if (size() == 0) {
      throw std::out_of_range("Deque::front() - deque is empty");
    }
    return *get_element_ptr(front_index);
  }

  /// @brief Returns a reference to the last element in the deque.
  T& back() {
    if (size() == 0) {
      throw std::out_of_range("Deque::back() - deque is empty");
    }
    return *get_element_ptr(back_index - 1);
  }

  /// @brief Returns a const reference to the last element in the deque.
  const T& back() const {
    if (size() == 0) {
      throw std::out_of_range("Deque::back() - deque is empty");
    }
    return *get_element_ptr(back_index - 1);
  }

  /// @brief Clear the deque
  void clear() {
    while (size() > 0)
      pop_back();
    // First destruct all the elements, then deallocate all the blocks.
    for (std::size_t i = front_index; i < back_index; ++i) {
      get_element_ptr(i)->~T();
    }
    for (auto& block : blocks) {
      // delete[] reinterpret_cast<char*>(block);  // ???
      free_block(block);
      block = nullptr;
    }

    blocks.clear();
    front_index = 0;
    back_index = 0;
  }

 protected:
  using Block = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

  /// @brief Allocates a new block of memory for storing elements.
  /// @return A pointer to the newly allocated block.
  Block* allocate_block() {
    // Allocate raw memory for a block, ensuring proper alignment for T
    return reinterpret_cast<Block*>(new char[kBlockSize * sizeof(Block)]);
  }

  /// @brief Frees the memory allocated for a block.
  /// @param block A pointer to the block to be freed.
  void free_block(Block* block) {
    delete[] reinterpret_cast<char*>(block);
  }

  /// @brief Ensures there is enough capacity at the back for one additional
  /// element.
  void ensure_back_capacity() {
    if (back_index == blocks.size() * kBlockSize) {
      blocks.push_back(allocate_block());
    }
  }

  /// @brief Ensures there is enough capacity at the back for `n` additional
  /// elements.
  /// @param n The number of additional elements required.
  void ensure_back_capacity(size_t n) {
    if (n == 1) {
      ensure_back_capacity();
      return;
    }

    size_t required_back_index =
        back_index + n;  // Calculate the required index for new elements
    size_t required_blocks = (required_back_index + kBlockSize - 1) /
                             kBlockSize;  // Total required blocks

    // If the required blocks exceed the current block size, resize
    if (required_blocks > blocks.size()) {
      blocks.reserve(required_blocks);
      size_t additional_blocks =
          required_blocks - blocks.size();  // Blocks to add at the back
      // Append new blocks to the back
      for (size_t i = 0; i < additional_blocks; ++i) {
        blocks.push_back(allocate_block());
      }
    }
  }

  /// @brief Ensures there is enough capacity at the front for one additional
  /// element.
  void ensure_front_capacity() {
    if (front_index == 0) {
      blocks.insert(blocks.begin(), allocate_block());
      front_index = kBlockSize;
      back_index += kBlockSize;
      return;
    }
  }

  /// @brief Ensures there is enough capacity at the front for `n` additional
  /// elements.
  /// @param n The number of additional elements required.
  void ensure_front_capacity(size_t n) {
    if (n == 1) {
      ensure_front_capacity();
    }

    size_t additional_blocks =
        (n < front_index) ? 0
                          : ((n - front_index + kBlockSize - 1) / kBlockSize);
    blocks.insert(blocks.begin(), additional_blocks, nullptr);
    for (size_t i = 0; i < additional_blocks; ++i) {
      if (!blocks[i]) {
        blocks[i] = allocate_block();
      }
    }
    // Adjust front_index and back_index to match the new layout
    front_index += additional_blocks * kBlockSize;
    back_index += additional_blocks * kBlockSize;
  }

  /// @brief Get a pointer to an element
  T* get_element_ptr(std::size_t index) {
    std::size_t block_index = index / kBlockSize;
    std::size_t block_offset = index % kBlockSize;
    return reinterpret_cast<T*>(&blocks[block_index][block_offset]);
  }

  /// @brief Get a pointer to an element (const version)
  const T* get_element_ptr(std::size_t index) const {
    std::size_t block_index = index / kBlockSize;
    std::size_t block_offset = index % kBlockSize;
    return reinterpret_cast<T*>(&blocks[block_index][block_offset]);
  }

 protected:
  std::vector<Block*> blocks;  ///< Vector managing block pointers
  std::size_t front_index;     ///< Logical front index
  std::size_t back_index;      ///< Logical back index
};

}  // namespace sk4slam

namespace std {

template <typename Deque>
struct iterator_traits<sk4slam::deque_internal::iterator<Deque>> {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = typename Deque::value_type;
  using difference_type = typename Deque::difference_type;
  using pointer = typename Deque::pointer;
  using reference = typename Deque::reference;
};

template <typename Deque>
struct iterator_traits<sk4slam::deque_internal::const_iterator<Deque>> {
  using iterator_category = std::random_access_iterator_tag;
  using value_type = typename Deque::value_type;
  using difference_type = typename Deque::difference_type;
  using pointer = typename Deque::const_pointer;
  using reference = typename Deque::const_reference;
};

template <typename Deque>
struct iterator_traits<sk4slam::deque_internal::reverse_iterator<Deque>> {
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = typename Deque::value_type;
  using difference_type = typename Deque::difference_type;
  using pointer = typename Deque::pointer;
  using reference = typename Deque::reference;
};

template <typename Deque>
struct iterator_traits<sk4slam::deque_internal::const_reverse_iterator<Deque>> {
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = typename Deque::value_type;
  using difference_type = typename Deque::difference_type;
  using pointer = typename Deque::const_pointer;
  using reference = typename Deque::const_reference;
};

}  // namespace std
