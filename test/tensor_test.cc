#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"

TEST(TensorTest, ConstructorInitializesShapeAndSize)
{
    std::vector<std::size_t> shape = {2, 3, 4};
    matrix_library::Tensor<int> tensor(shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ((tensor[{1, 2, 3}]), 0);
}

TEST(TensorTest, SetDataCopyAndFill)
{
    std::vector<std::size_t> shape = {2, 2};
    matrix_library::Tensor<int> tensor(shape);

    std::vector<int> vals = {1, 2, 3, 4};
    tensor.set_data(vals);

    EXPECT_EQ((tensor[{0, 0}]), 1);
    EXPECT_EQ((tensor[{0, 1}]), 2);
    EXPECT_EQ((tensor[{1, 0}]), 3);
    EXPECT_EQ((tensor[{1, 1}]), 4);

    tensor.fill(7);
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        size_t r = i / 2;
        size_t c = i % 2;
        EXPECT_EQ((tensor[{r, c}]), 7);
    }
}

TEST(TensorTest, SetDataMoveAndInvalid)
{
    std::vector<std::size_t> shape = {3};
    matrix_library::Tensor<int> tensor(shape);

    std::vector<int> vals = {10, 11, 12};
    tensor.set_data(std::move(vals));

    EXPECT_EQ((tensor[{0}]), 10);
    EXPECT_EQ((tensor[{1}]), 11);
    EXPECT_EQ((tensor[{2}]), 12);

    std::vector<int> bad = {1, 2};
    EXPECT_THROW(tensor.set_data(bad), std::invalid_argument);
}
