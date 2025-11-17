#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"
#include "operations/apply.h"

TEST(OperationsApplyTest, SquareEachElementIntInPlace)
{
    matrix_library::Tensor<int> a(std::vector<std::size_t>{2, 3});
    a.set_data(std::vector<int>{1, -2, 3, -4, 5, -6});

    matrix_library::operations::apply_inplace(a, [](int x) { return x * x; });

    EXPECT_EQ(a.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ((a[{0, 0}]), 1);
    EXPECT_EQ((a[{0, 1}]), 4);
    EXPECT_EQ((a[{0, 2}]), 9);
    EXPECT_EQ((a[{1, 0}]), 16);
    EXPECT_EQ((a[{1, 1}]), 25);
    EXPECT_EQ((a[{1, 2}]), 36);
}

TEST(OperationsApplyTest, IncrementEachElementInPlace1D)
{
    matrix_library::Tensor<int> a(std::vector<std::size_t>{3});
    a.set_data(std::vector<int>{2, 4, 6});

    matrix_library::operations::apply_inplace(a, [](int x) { return x + 1; });

    EXPECT_EQ(a.shape(), (std::vector<std::size_t>{3}));
    EXPECT_EQ((a[{0}]), 3);
    EXPECT_EQ((a[{1}]), 5);
    EXPECT_EQ((a[{2}]), 7);
}

TEST(OperationsApplyTest, WorksFor3DTensor)
{
    matrix_library::Tensor<int> a(std::vector<std::size_t>{2, 2, 2});
    a.set_data(std::vector<int>{0,1,2,3,4,5,6,7});

    matrix_library::operations::apply_inplace(a, [](int x) { return x + 10; });

    EXPECT_EQ(a.shape(), (std::vector<std::size_t>{2, 2, 2}));
    EXPECT_EQ((a[{0,0,0}]), 10);
    EXPECT_EQ((a[{0,0,1}]), 11);
    EXPECT_EQ((a[{0,1,0}]), 12);
    EXPECT_EQ((a[{0,1,1}]), 13);
    EXPECT_EQ((a[{1,0,0}]), 14);
    EXPECT_EQ((a[{1,0,1}]), 15);
    EXPECT_EQ((a[{1,1,0}]), 16);
    EXPECT_EQ((a[{1,1,1}]), 17);
}

TEST(OperationsApplyTest, EmptyTensorNoOp)
{
    matrix_library::Tensor<double> a(std::vector<std::size_t>{});
    EXPECT_NO_THROW(matrix_library::operations::apply_inplace(a, [](double x) { return x + 1.0; }));
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.shape(), std::vector<std::size_t>{});
}
