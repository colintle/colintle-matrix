#include <gtest/gtest.h>

TEST(OperationsTest, BasicAssertions) {
    EXPECT_STREQ("hello", "hello");
    EXPECT_EQ(42, 42);
}