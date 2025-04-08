#include "sk4slam_basic/unique_id.h"

#include <set>
#include <unordered_set>

#include "sk4slam_basic/logging.h"
#include "sk4slam_basic/ut_entrypoint.h"

TEST(TestUniqueId, String) {
  sk4slam::UniqueId id1, id2;
  std::cout << "id1: " << id1 << std::endl;
  std::cout << "id2: " << id2 << std::endl;
  ASSERT_NE(id1, id2);
  ASSERT_EQ(id1.hexString(), id1.hexString());
  ASSERT_EQ(id2.hexString(), id2.hexString());
  ASSERT_NE(id1.hexString(), id2.hexString());
  ASSERT_EQ(id1.hexString().length(), 32);
  ASSERT_EQ(id2.hexString().length(), 32);
  id2.fromHexString(id1.hexString());
  ASSERT_EQ(id1, id2);
}

TEST(TestUniqueId, Key) {
  std::set<sk4slam::UniqueId> ordered_set;
  std::unordered_set<sk4slam::UniqueId> unordered_set;
  sk4slam::UniqueId id1, id2;
  std::cout << "id1: " << id1 << std::endl;
  std::cout << "id2: " << id2 << std::endl;
  ordered_set.insert(id1);
  ordered_set.insert(id2);
  unordered_set.insert(id1);
  unordered_set.insert(id2);
  ASSERT_EQ(ordered_set.size(), 2);
  ASSERT_EQ(unordered_set.size(), 2);
  ASSERT_EQ(ordered_set.count(id1), 1);
  ASSERT_EQ(ordered_set.count(id2), 1);
  ASSERT_EQ(unordered_set.count(id1), 1);
  ASSERT_EQ(unordered_set.count(id2), 1);
}

SK4SLAM_UNITTEST_ENTRYPOINT
