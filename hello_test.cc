#include <gtest/gtest.h>

TEST(HelloTest, BasicAssertions) {
    EXPECT_STREQ("hello", "hello");
    EXPECT_EQ(42, 42);
}