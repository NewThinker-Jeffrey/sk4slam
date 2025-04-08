#include "sk4slam_cpp/deque.h"

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/string_helper.h"
#include "sk4slam_basic/ut_entrypoint.h"

using namespace sk4slam;  // NOLINT

// TestableDeque: A subclass of Deque with additional methods for testing
template <typename T, std::size_t BlockSize>
class TestableDeque : public Deque<T, BlockSize> {
 public:
  using Deque<T, BlockSize>::blocks;          // Expose blocks
  using Deque<T, BlockSize>::front_index;     // Expose front_index
  using Deque<T, BlockSize>::back_index;      // Expose back_index
  using Deque<T, BlockSize>::allocate_block;  // Expose allocate_block

  // Friend the test framework to allow direct access for assertions
  template <typename U, std::size_t BS>
  friend class TestableDeque;
};

// Test fixture for the Deque class
template <typename T>
class DequeTest : public ::testing::Test {
 protected:
  Deque<T, 4> deque;  // Test instance with block size 4
};

using TestTypes = ::testing::Types<int, double>;
TYPED_TEST_SUITE(DequeTest, TestTypes);

// Test: Default construction
TYPED_TEST(DequeTest, DefaultConstructor) {
  EXPECT_EQ(this->deque.size(), 0);
}

// Test: Push and access elements
TYPED_TEST(DequeTest, PushAndAccess) {
  this->deque.push_back(1);
  this->deque.push_back(2);
  this->deque.push_front(0);

  EXPECT_EQ(this->deque.size(), 3);
  EXPECT_EQ(this->deque[0], 0);
  EXPECT_EQ(this->deque[1], 1);
  EXPECT_EQ(this->deque[2], 2);
}

// Test: Emplace elements
TYPED_TEST(DequeTest, EmplaceElements) {
  this->deque.emplace_back(42);
  this->deque.emplace_front(24);

  EXPECT_EQ(this->deque.size(), 2);
  EXPECT_EQ(this->deque[0], 24);
  EXPECT_EQ(this->deque[1], 42);
}

// Test: Insert elements
TYPED_TEST(DequeTest, InsertElements) {
  this->deque.push_back(1);
  this->deque.push_back(3);
  this->deque.insert(this->deque.begin() + 1, 2);

  EXPECT_EQ(this->deque.size(), 3);
  EXPECT_EQ(this->deque[0], 1);
  EXPECT_EQ(this->deque[1], 2);
  EXPECT_EQ(this->deque[2], 3);
}

// Test: Erase elements
TYPED_TEST(DequeTest, EraseElements) {
  this->deque.push_back(1);
  this->deque.push_back(2);
  this->deque.push_back(3);

  this->deque.erase(this->deque.begin() + 1);  // Remove element at index 1
  EXPECT_EQ(this->deque.size(), 2);
  EXPECT_EQ(this->deque[0], 1);
  EXPECT_EQ(this->deque[1], 3);

  this->deque.shrink_to_fit();
  EXPECT_EQ(this->deque.size(), 2);
  EXPECT_EQ(this->deque[0], 1);
  EXPECT_EQ(this->deque[1], 3);
}

// Test: Clear the deque
TYPED_TEST(DequeTest, Clear) {
  this->deque.push_back(1);
  this->deque.push_back(2);

  this->deque.clear();
  EXPECT_EQ(this->deque.size(), 0);
}

// Test: Copy constructor
TYPED_TEST(DequeTest, CopyConstructor) {
  this->deque.push_back(1);
  this->deque.push_back(2);

  Deque<TypeParam, 4> copy(this->deque);
  EXPECT_EQ(copy.size(), 2);
  EXPECT_EQ(copy[0], 1);
  EXPECT_EQ(copy[1], 2);
}

// Test: Copy assignment operator
TYPED_TEST(DequeTest, CopyAssignment) {
  this->deque.push_back(1);
  this->deque.push_back(2);

  Deque<TypeParam, 4> copy;
  copy = this->deque;

  EXPECT_EQ(copy.size(), 2);
  EXPECT_EQ(copy[0], 1);
  EXPECT_EQ(copy[1], 2);
}

// Test: Move constructor
TYPED_TEST(DequeTest, MoveConstructor) {
  this->deque.push_back(1);
  this->deque.push_back(2);

  Deque<TypeParam, 4> moved(std::move(this->deque));
  EXPECT_EQ(moved.size(), 2);
  EXPECT_EQ(moved[0], 1);
  EXPECT_EQ(moved[1], 2);
  EXPECT_EQ(this->deque.size(), 0);  // Source deque should be empty
}

// Test: Move assignment operator
TYPED_TEST(DequeTest, MoveAssignment) {
  this->deque.push_back(1);
  this->deque.push_back(2);

  Deque<TypeParam, 4> moved;
  moved = std::move(this->deque);

  EXPECT_EQ(moved.size(), 2);
  EXPECT_EQ(moved[0], 1);
  EXPECT_EQ(moved[1], 2);
  EXPECT_EQ(this->deque.size(), 0);  // Source deque should be empty
}

// Test: Out-of-bounds access
TYPED_TEST(DequeTest, OutOfBoundsAccess) {
  this->deque.push_back(1);

  EXPECT_THROW(this->deque[1], std::out_of_range);
  EXPECT_THROW(this->deque[100], std::out_of_range);
}

// Test: Push elements with large block size
TYPED_TEST(DequeTest, LargeBlockSize) {
  Deque<TypeParam, 1024> largeDeque;  // Block size 1024
  largeDeque.push_back(1);
  largeDeque.push_back(2);
  EXPECT_EQ(largeDeque.size(), 2);
  EXPECT_EQ(largeDeque[0], 1);
  EXPECT_EQ(largeDeque[1], 2);
}

// Test: Pop elements from front and back
TYPED_TEST(DequeTest, PopElements) {
  this->deque.push_back(1);
  this->deque.push_back(2);
  this->deque.push_back(3);

  this->deque.pop_front();
  EXPECT_EQ(this->deque.size(), 2);
  EXPECT_EQ(this->deque[0], 2);

  this->deque.pop_back();
  EXPECT_EQ(this->deque.size(), 1);
  EXPECT_EQ(this->deque[0], 2);

  this->deque.shrink_to_fit();
  EXPECT_EQ(this->deque.size(), 1);
  EXPECT_EQ(this->deque[0], 2);
}

// Test trim_unused_blocks()
TEST(DequeTest, TrimUnusedBlocks) {
  TestableDeque<int, 4> dq;
  EXPECT_EQ(dq.blocks.size(), 0);

  // Add elements
  for (int i = 0; i < 20; ++i) {
    dq.push_back(i);
    ASSERT_EQ(dq.blocks.size(), i / 4 + 1)
        << "Expected " << i / 4 + 1 << " blocks when after element " << i;
  }
  EXPECT_EQ(dq.blocks.size(), 5);

  // Remove some elements
  for (int i = 0; i < 10; ++i) {
    dq.pop_front();
  }

  // Reserve additional blocks for testing
  dq.blocks.push_back(dq.allocate_block());
  dq.blocks.push_back(dq.allocate_block());

  // Before trimming
  EXPECT_EQ(dq.blocks.size(), 7);  // 7 blocks should be allocated
  EXPECT_EQ(dq.front_index, 10);
  EXPECT_EQ(dq.back_index, 20);

  // Perform trim_unused_blocks
  dq.trim_unused_blocks(1, 1);

  // After trimming
  EXPECT_EQ(dq.blocks.size(), 5);  // Only 4 blocks should remain
  EXPECT_EQ(dq.front_index, 6);
  EXPECT_EQ(dq.back_index, 16);
}

// Test trim_to_optimal()
TEST(DequeTest, TrimToOptimal) {
  TestableDeque<int, 4> dq;

  // Add elements
  for (int i = 0; i < 50; ++i) {
    dq.push_back(i);
  }

  // Remove some elements
  for (int i = 0; i < 20; ++i) {
    dq.pop_front();
  }

  // Before trimming
  EXPECT_EQ(dq.blocks.size(), 13);  // 13 blocks should be allocated
  EXPECT_EQ(dq.front_index, 20);
  EXPECT_EQ(dq.back_index, 50);
  EXPECT_EQ(dq.size(), 30);

  // Perform trim_to_optimal
  dq.trim_to_optimal();

  // After trimming
  EXPECT_EQ(
      dq.blocks.size(), 10);  // Only 10 blocks should remain (3 blocks will be
                              // trimmed from the front, 0 from the back)
  EXPECT_EQ(dq.front_index, 8);  // Adjusted to the first block
  EXPECT_EQ(dq.back_index, 38);  // Correctly adjusted

  // Reserve additional blocks for testing
  dq.blocks.push_back(dq.allocate_block());
  dq.blocks.push_back(dq.allocate_block());
  dq.blocks.push_back(dq.allocate_block());

  EXPECT_EQ(dq.blocks.size(), 13);
  EXPECT_EQ(dq.front_index, 8);
  EXPECT_EQ(dq.back_index, 38);

  // Perform trim_to_optimal again
  dq.trim_to_optimal();

  // After trimming again
  EXPECT_EQ(dq.blocks.size(), 12);  // Only 12 blocks should remain (1 block
                                    // will be trimmed from the back)
  EXPECT_EQ(dq.front_index, 8);
  EXPECT_EQ(dq.back_index, 38);
}

// Verify data integrity after trimming
TEST(DequeTest, DataIntegrityAfterTrim) {
  TestableDeque<int, 4> dq;

  // Add elements
  for (int i = 0; i < 20; ++i) {
    dq.push_back(i);
  }

  // Remove some elements
  for (int i = 0; i < 8; ++i) {
    dq.pop_front();
  }

  // Save original data
  std::vector<int> original_data;
  for (size_t i = 0; i < dq.size(); ++i) {
    original_data.push_back(dq[i]);
  }

  // Perform trim_to_optimal
  dq.trim_to_optimal();

  // Verify size remains the same
  EXPECT_EQ(dq.size(), original_data.size());

  // Verify data integrity
  for (size_t i = 0; i < dq.size(); ++i) {
    EXPECT_EQ(dq[i], original_data[i]);
  }
}

// Test edge cases
TEST(DequeTest, EdgeCaseTrimMethods) {
  TestableDeque<int, 4> dq;

  // Empty deque
  dq.trim_to_optimal();
  EXPECT_EQ(dq.blocks.size(), 0);

  dq.trim_unused_blocks(1, 1);
  EXPECT_EQ(dq.blocks.size(), 0);

  // Single element deque
  dq.push_back(42);
  dq.trim_to_optimal();
  EXPECT_EQ(dq.blocks.size(), 1);
  EXPECT_EQ(dq[0], 42);

  dq.trim_unused_blocks(0, 0);
  EXPECT_EQ(dq.blocks.size(), 1);
  EXPECT_EQ(dq[0], 42);
}

// Test begin() and end() for iterator
TEST(DequeTest, IteratorTest) {
  TestableDeque<int, 4> dq;

  // Add elements to deque
  for (int i = 0; i < 10; ++i) {
    dq.push_back(i);
  }

  // Use iterator with begin() and end()
  std::vector<int> data;
  for (auto it = dq.begin(); it != dq.end(); ++it) {
    data.push_back(*it);
  }

  // Check if the data is in correct order
  std::vector<int> expected_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_EQ(data, expected_data);
}

// Test begin() and end() for const_iterator
TEST(DequeTest, ConstIteratorTest) {
  TestableDeque<int, 4> mutable_dq;

  // Add elements to deque
  for (int i = 0; i < 10; ++i) {
    mutable_dq.push_back(i);
  }

  const TestableDeque<int, 4> dq = mutable_dq;

  // Use const_iterator with begin() and end()
  std::vector<int> data;
  for (auto it = dq.begin(); it != dq.end(); ++it) {
    data.push_back(*it);
  }

  // Check if the data is in correct order
  std::vector<int> expected_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_EQ(data, expected_data);
}

// Test begin() and end() on empty deque
TEST(DequeTest, IteratorEmptyDeque) {
  TestableDeque<int, 4> dq;

  // Test begin() and end() on an empty deque
  auto b = dq.begin();
  auto e = dq.end();

  // Ensure begin() == end() on empty deque
  EXPECT_EQ(b, e);
}

// Test rbegin() and rend() for reverse_iterator
TEST(DequeTest, ReverseIteratorTest) {
  TestableDeque<int, 4> dq;

  // Add elements to deque
  for (int i = 0; i < 10; ++i) {
    dq.push_back(i);
  }

  // Use reverse_iterator with rbegin and rend
  std::vector<int> reversed_data;
  for (auto it = dq.rbegin(); it != dq.rend(); ++it) {
    reversed_data.push_back(*it);
  }

  // Check if the data is in reverse order
  std::vector<int> expected_data = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  EXPECT_EQ(reversed_data, expected_data);
}

// Test rbegin() and rend() for const_reverse_iterator
TEST(DequeTest, ConstReverseIteratorTest) {
  TestableDeque<int, 4> mutable_dq;

  // Add elements to deque
  for (int i = 0; i < 10; ++i) {
    mutable_dq.push_back(i);
  }

  const TestableDeque<int, 4> dq = mutable_dq;

  // Use const_reverse_iterator with rbegin and rend
  std::vector<int> reversed_data;
  for (auto it = dq.rbegin(); it != dq.rend(); ++it) {
    reversed_data.push_back(*it);
  }

  // Check if the data is in reverse order
  std::vector<int> expected_data = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  EXPECT_EQ(reversed_data, expected_data);
}

// Test rbegin() and rend() on empty deque
TEST(DequeTest, ReverseIteratorEmptyDeque) {
  TestableDeque<int, 4> dq;

  // Test rbegin() and rend() on an empty deque
  auto rb = dq.rbegin();
  auto re = dq.rend();

  // Ensure rbegin() == rend() on empty deque
  EXPECT_EQ(rb, re);
}

SK4SLAM_UNITTEST_ENTRYPOINT
