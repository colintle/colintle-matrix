#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"
#include "operations/transpose.h"

TEST(OperationsTransposeTest, SimpleTranspose2D)
{
    matrix_library::Tensor<int> a(std::vector<std::size_t>{2, 3});
    a.set_data(std::vector<int>{1, 2, 3, 4, 5, 6}); // [[1,2,3],[4,5,6]]

    auto t = matrix_library::operations::transpose(a);
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{3, 2}));
    EXPECT_EQ((t[{0, 0}]), 1);
    EXPECT_EQ((t[{0, 1}]), 4);
    EXPECT_EQ((t[{1, 0}]), 2);
    EXPECT_EQ((t[{1, 1}]), 5);
    EXPECT_EQ((t[{2, 0}]), 3);
    EXPECT_EQ((t[{2, 1}]), 6);
}

TEST(OperationsTransposeTest, Transpose1D)
{
    matrix_library::Tensor<int> v(std::vector<std::size_t>{3});
    v.set_data(std::vector<int>{7, 8, 9});

    auto t = matrix_library::operations::transpose(v);
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{3, 1}));
    EXPECT_EQ((t[{0, 0}]), 7);
    EXPECT_EQ((t[{1, 0}]), 8);
    EXPECT_EQ((t[{2, 0}]), 9);
}

TEST(OperationsTransposeTest, EmptyTensor)
{
    matrix_library::Tensor<int> a(std::vector<std::size_t>{});
    auto t = matrix_library::operations::transpose(a);
    EXPECT_EQ(t.shape(), std::vector<std::size_t>{});
}
