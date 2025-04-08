#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/reflection.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/template_helper.h"

namespace sk4slam {

namespace binary_search_internal {
enum class SearchMode {
  /// Return any exact match
  Any,

  /// Return the first exact match
  First,

  /// Return the last exact match
  Last,

  /// Return both the first and last exact matches
  FirstAndLast,

  /// Returns an iterator pointing to the first element in the container whose
  /// key is not considered to go before the target (i.e., either it is
  /// equivalent or goes after).
  /// e.g. (when the container is sorted in ascending order, return an iterator
  ///       pointing to the first element in the container whose key is equal to
  ///       or greater than the target):
  ///      return 0 if target <= arr[0],
  ///      return arr.size() if target > arr[arr.size() - 1],
  ///      return i + 1 if arr[i] < target <= arr[i+1]
  LowerBound,

  /// Returns an iterator pointing to the first element in the container whose
  /// key is considered to go after the target.
  /// e.g. (when the container is sorted in ascending order, return an iterator
  ///       pointing to the first element in the container whose key is greater
  ///       than the target):
  ///      return 0 if target < arr[0],
  ///      return arr.size() if target >= arr[arr.size() - 1],
  ///      return i + 1 if arr[i] <= target < arr[i+1]
  UpperBound
};

static constexpr SearchMode Any = SearchMode::Any;
static constexpr SearchMode First = SearchMode::First;
static constexpr SearchMode Last = SearchMode::Last;
static constexpr SearchMode FirstAndLast = SearchMode::FirstAndLast;
static constexpr SearchMode LowerBound = SearchMode::LowerBound;
static constexpr SearchMode UpperBound = SearchMode::UpperBound;

template <SearchMode Mode>
using ReturnType = std::conditional_t<
    Mode == FirstAndLast, std::optional<std::pair<size_t, size_t>>,
    std::optional<size_t>>;

template <SearchMode Mode, typename Iterator>
using IterReturnType = std::conditional_t<
    Mode == FirstAndLast, std::optional<std::pair<Iterator, Iterator>>,
    std::optional<Iterator>>;

template <typename T, typename Key>
struct DefaultKeyConvert {
  Key operator()(const T& t) const {
    // Check whether T is convertible to Key
    if constexpr (std::is_convertible_v<T, Key>) {
      return static_cast<Key>(t);
    } else {
      throw std::runtime_error(formatStr(
          "In BinarySearch: DefaultKeyConvert can't convert type \"%s\" "
          "to the Key type \"%s\"!",
          classname<T>(), classname<Key>()));
    }
  }
};

/// @brief  Traits for container types that can be used with binarySearch().
///
/// The default implementation assumes that the container is a std::vector like
/// type. If the container is not std::vector like, you can specialize this
/// template for your container.
template <typename Container>
struct container_traits {
  using value_type = typename Container::value_type;
  static const value_type& at(const Container& c, size_t i) {
    return c[i];
  }
  static size_t size(const Container& c) {
    return c.size();
  }
};

/// @brief  Traits for iterator types that can be used with binarySearch().
///
/// The default implementation assumes that the iterator is a
/// std::vector::iterator like type. If the iterator is not
/// std::vector::iterator like, you can specialize this template for your
/// iterator.
template <typename Iterator>
struct iterator_traits {
  using value_type = typename std::iterator_traits<Iterator>::value_type;
  static const value_type& get(const Iterator& it) {
    return *it;
  }
  static int distance(const Iterator& begin, const Iterator& end) {
    return std::distance(begin, end);
  }
  static Iterator advance(const Iterator& it, int n) {
    return it + n;
  }
  static bool same(const Iterator& it1, const Iterator& it2) {
    return it1 == it2;
  }
};

}  // namespace binary_search_internal

template <
    binary_search_internal::SearchMode Mode, typename Key, typename Container,
    typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::container_traits<
            Container>::value_type,
        Key>>
binary_search_internal::ReturnType<Mode> binarySearch(
    const Container& arr, const Key& target, const Compare& cmp = Compare(),
    const KeyFunc& key_func = KeyFunc());

template <
    binary_search_internal::SearchMode Mode, typename Key, typename Iterator,
    typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::iterator_traits<Iterator>::value_type,
        Key>>
binary_search_internal::IterReturnType<Mode, Iterator> binarySearch(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp = Compare(), const KeyFunc& key_func = KeyFunc());

/// @name Search for exact match
/// @{

template <
    typename Key, typename Container, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::container_traits<
            Container>::value_type,
        Key>>
inline std::optional<size_t> binarySearchAny(
    const Container& arr, const Key& target, const Compare& cmp = Compare(),
    const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::Any>(arr, target, cmp, key_func);
}

template <
    typename Key, typename Iterator, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::iterator_traits<Iterator>::value_type,
        Key>>
inline std::optional<Iterator> binarySearchAny(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp = Compare(), const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::Any>(
      begin, end, target, cmp, key_func);
}

template <
    typename Key, typename Container, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::container_traits<
            Container>::value_type,
        Key>>
inline std::optional<size_t> binarySearchFirst(
    const Container& arr, const Key& target, const Compare& cmp = Compare(),
    const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::First>(
      arr, target, cmp, key_func);
}

template <
    typename Key, typename Iterator, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::iterator_traits<Iterator>::value_type,
        Key>>
inline std::optional<Iterator> binarySearchFirst(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp = Compare(), const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::First>(
      begin, end, target, cmp, key_func);
}

template <
    typename Key, typename Container, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::container_traits<
            Container>::value_type,
        Key>>
inline std::optional<size_t> binarySearchLast(
    const Container& arr, const Key& target, const Compare& cmp = Compare(),
    const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::Last>(arr, target, cmp, key_func);
}

template <
    typename Key, typename Iterator, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::iterator_traits<Iterator>::value_type,
        Key>>
inline std::optional<Iterator> binarySearchLast(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp = Compare(), const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::Last>(
      begin, end, target, cmp, key_func);
}

template <
    typename Key, typename Container, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::container_traits<
            Container>::value_type,
        Key>>
inline std::optional<std::pair<size_t, size_t>> binarySearchFirstAndLast(
    const Container& arr, const Key& target, const Compare& cmp = Compare(),
    const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::FirstAndLast>(
      arr, target, cmp, key_func);
}

template <
    typename Key, typename Iterator, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::iterator_traits<Iterator>::value_type,
        Key>>
inline std::optional<std::pair<Iterator, Iterator>> binarySearchFirstAndLast(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp = Compare(), const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::FirstAndLast>(
      begin, end, target, cmp, key_func);
}

/// @}
/// @name Search for boundary
/// @{

template <
    typename Key, typename Container, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::container_traits<
            Container>::value_type,
        Key>>
inline std::optional<size_t> binarySearchLowerBound(
    const Container& arr, const Key& target, const Compare& cmp = Compare(),
    const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::LowerBound>(
      arr, target, cmp, key_func);
}

template <
    typename Key, typename Iterator, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::iterator_traits<Iterator>::value_type,
        Key>>
inline std::optional<Iterator> binarySearchLowerBound(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp = Compare(), const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::LowerBound>(
      begin, end, target, cmp, key_func);
}

template <
    typename Key, typename Container, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::container_traits<
            Container>::value_type,
        Key>>
inline std::optional<size_t> binarySearchUpperBound(
    const Container& arr, const Key& target, const Compare& cmp = Compare(),
    const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::UpperBound>(
      arr, target, cmp, key_func);
}

template <
    typename Key, typename Iterator, typename Compare = std::less<Key>,
    typename KeyFunc = binary_search_internal::DefaultKeyConvert<
        typename binary_search_internal::iterator_traits<Iterator>::value_type,
        Key>>
inline std::optional<Iterator> binarySearchUpperBound(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp = Compare(), const KeyFunc& key_func = KeyFunc()) {
  return binarySearch<binary_search_internal::UpperBound>(
      begin, end, target, cmp, key_func);
}

/// @}

///////// Implementation //////////

template <
    binary_search_internal::SearchMode Mode, typename Key, typename Container,
    typename Compare, typename KeyFunc>
binary_search_internal::ReturnType<Mode> binarySearch(
    const Container& arr, const Key& target, const Compare& cmp,
    const KeyFunc& key_func) {
  using namespace binary_search_internal;  // NOLINT
  using traits = container_traits<Container>;

  if (traits::size(arr) == 0) {
    if constexpr (Mode == LowerBound || Mode == UpperBound) {
      return 0;
    } else {
      return std::nullopt;
    }
  }

  size_t left = 0;
  size_t right = traits::size(arr);

  while (left < right) {
    size_t mid = left + (right - left) / 2;
    if (cmp(key_func(traits::at(arr, mid)), target)) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  auto equals = [&cmp](const Key& a, const Key& b) {
    return !cmp(a, b) && !cmp(b, a);
  };

  if constexpr (Mode == LowerBound) {
    // Return the lower bound.
    // e.g. (when the container is sorted in ascending order, return an iterator
    //       pointing to the first element in the container whose key is equal
    //       to or greater than the target):
    //      return 0 if target <= arr[0],
    //      return arr.size() if target > arr[arr.size() - 1],
    //      return i + 1 if arr[i] < target <= arr[i+1]
    while (left > 0 && equals(target, key_func(traits::at(arr, left - 1)))) {
      --left;
    }
    if (left == traits::size(arr) ||
        !cmp(key_func(traits::at(arr, left)), target)) {
      return left;
    } else {
      ASSERT(cmp(key_func(traits::at(arr, left)), target));
      return left + 1;
    }
  } else if constexpr (Mode == UpperBound) {
    // Return the upper bound.
    // e.g. (when the container is sorted in ascending order, return an iterator
    //       pointing to the first element in the container whose key is greater
    //       than the target):
    //      return 0 if target < arr[0],
    //      return arr.size() if target >= arr[arr.size() - 1],
    //      return i + 1 if arr[i] <= target < arr[i+1]
    size_t last = left;
    while (last + 1 < traits::size(arr) &&
           equals(key_func(traits::at(arr, last + 1)), target)) {
      ++last;
    }

    if (last == traits::size(arr) ||
        cmp(target, key_func(traits::at(arr, last)))) {
      return last;
    } else {
      ASSERT(!cmp(target, key_func(traits::at(arr, last))));
      return last + 1;
    }
  } else {
    if (left < traits::size(arr) &&
        equals(key_func(traits::at(arr, left)), target)) {
      if constexpr (Mode == Any) {
        return left;
      } else if constexpr (Mode == First) {
        while (left > 0 &&
               equals(key_func(traits::at(arr, left - 1)), target)) {
          --left;
        }
        return left;
      } else if constexpr (Mode == Last) {
        size_t last = left;
        while (last + 1 < traits::size(arr) &&
               equals(key_func(traits::at(arr, last + 1)), target)) {
          ++last;
        }
        return last;
      } else if constexpr (Mode == FirstAndLast) {
        size_t first = left;
        size_t last = left;
        while (first > 0 &&
               equals(key_func(traits::at(arr, first - 1)), target)) {
          --first;
        }
        while (last + 1 < traits::size(arr) &&
               equals(key_func(traits::at(arr, last + 1)), target)) {
          ++last;
        }
        return std::make_pair(first, last);
      }
    }
  }

  return std::nullopt;
}

template <
    binary_search_internal::SearchMode Mode, typename Key, typename Iterator,
    typename Compare, typename KeyFunc>
binary_search_internal::IterReturnType<Mode, Iterator> binarySearch(
    const Iterator& begin, const Iterator& end, const Key& target,
    const Compare& cmp, const KeyFunc& key_func) {
  using namespace binary_search_internal;  // NOLINT
  using traits = iterator_traits<Iterator>;

  if (traits::same(begin, end)) {
    if constexpr (Mode == LowerBound || Mode == UpperBound) {
      return end;
    } else {
      return std::nullopt;
    }
  }

  Iterator left = begin;
  Iterator right = end;

  while (left < right) {
    Iterator mid = traits::advance(left, traits::distance(left, right) / 2);
    if (cmp(key_func(traits::get(mid)), target)) {
      left = traits::advance(mid, 1);
    } else {
      right = mid;
    }
  }

  auto equals = [&cmp](const Key& a, const Key& b) {
    return !cmp(a, b) && !cmp(b, a);
  };

  if constexpr (Mode == LowerBound) {
    // Return the lower bound.
    // e.g. (when the container is sorted in ascending order, return an iterator
    //       pointing to the first element in the container whose key is equal
    //       to or greater than the target):
    //      return 0 if target <= arr[0],
    //      return arr.size() if target > arr[arr.size() - 1],
    //      return i + 1 if arr[i] < target <= arr[i+1]
    while (left != begin &&
           equals(target, key_func(traits::get(traits::advance(left, -1))))) {
      left = traits::advance(left, -1);
    }

    if (traits::same(left, end) || !cmp(key_func(traits::get(left)), target)) {
      return left;
    } else {
      ASSERT(cmp(key_func(traits::get(left)), target));
      return traits::advance(left, 1);
    }
  } else if constexpr (Mode == UpperBound) {
    // Return the upper bound.
    // e.g. (when the container is sorted in ascending order, return an iterator
    //       pointing to the first element in the container whose key is greater
    //       than the target):
    //      return 0 if target < arr[0],
    //      return arr.size() if target >= arr[arr.size() - 1],
    //      return i + 1 if arr[i] <= target < arr[i+1]
    Iterator last = left;
    if (last != end) {
      while ((traits::advance(last, 1)) != end &&
             equals(key_func(traits::get(traits::advance(last, 1))), target)) {
        last = traits::advance(last, 1);
      }
    }

    if (traits::same(last, end) || cmp(target, key_func(traits::get(last)))) {
      return last;
    } else {
      ASSERT(!cmp(target, key_func(traits::get(last))));
      return traits::advance(last, 1);
    }
  } else {
    if (left != end && equals(key_func(traits::get(left)), target)) {
      if constexpr (Mode == Any) {
        return left;
      } else if constexpr (Mode == First) {
        while (
            left != begin &&
            equals(key_func(traits::get(traits::advance(left, -1))), target)) {
          left = traits::advance(left, -1);
        }
        return left;
      } else if constexpr (Mode == Last) {
        Iterator last = left;
        while (
            (traits::advance(last, 1)) != end &&
            equals(key_func(traits::get(traits::advance(last, 1))), target)) {
          last = traits::advance(last, 1);
        }
        return last;
      } else if constexpr (Mode == FirstAndLast) {
        Iterator first = left;
        Iterator last = left;
        while (
            first != begin &&
            equals(key_func(traits::get(traits::advance(first, -1))), target)) {
          first = traits::advance(first, -1);
        }
        while (
            (traits::advance(last, 1)) != end &&
            equals(key_func(traits::get(traits::advance(last, 1))), target)) {
          last = traits::advance(last, 1);
        }
        return std::make_pair(first, last);
      }
    }
  }

  return std::nullopt;
}

}  // namespace sk4slam
