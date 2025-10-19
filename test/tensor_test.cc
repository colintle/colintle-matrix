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

TEST(TensorTest, EmptyShapeBehavior)
{
    std::vector<std::size_t> shape = {};
    matrix_library::Tensor<int> tensor(shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 0);

    // Filling an empty tensor should be a no-op and not throw
    EXPECT_NO_THROW(tensor.fill(5));
}

TEST(TensorTest, OutOfRangeAccessThrows)
{
    std::vector<std::size_t> shape = {2, 2};
    matrix_library::Tensor<int> tensor(shape);

    // Expect out_of_range when index dimensionality mismatches
    std::vector<std::size_t> bad_dim = {0};
    EXPECT_THROW((void)tensor[bad_dim], std::out_of_range);

    // Expect out_of_range when index is too large
    std::vector<std::size_t> too_large1 = {2, 0};
    std::vector<std::size_t> too_large2 = {0, 3};
    EXPECT_THROW((void)tensor[too_large1], std::out_of_range);
    EXPECT_THROW((void)tensor[too_large2], std::out_of_range);
}

TEST(TensorTest, ConstAccessAnd3DIndexing)
{
    std::vector<std::size_t> shape = {2, 3, 4};
    matrix_library::Tensor<int> tensor(shape);

    std::vector<int> vals(24);
    for (size_t i = 0; i < vals.size(); ++i) vals[i] = static_cast<int>(i + 1);
    tensor.set_data(vals);

    const matrix_library::Tensor<int> &ct = tensor;
    std::vector<std::size_t> idx000 = {0, 0, 0};
    std::vector<std::size_t> idx123 = {1, 2, 3};
    std::vector<std::size_t> idx100 = {1, 0, 0};
    std::vector<std::size_t> idx010 = {0, 1, 0};

    EXPECT_EQ((ct[idx000]), 1);
    EXPECT_EQ((ct[idx123]), 24);

    EXPECT_EQ((ct[idx100]), 13);
    EXPECT_EQ((ct[idx010]), 5);
}
