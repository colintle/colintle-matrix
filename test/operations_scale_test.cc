#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"
#include "operations/scale.h"

TEST(OperationsScaleTest, SimpleScale2D)
{
    std::vector<std::size_t> shape = {2, 2};
    matrix_library::Tensor<int> a(shape);

    a.set_data(std::vector<int>{1, 2, 3, 4});

    auto c = matrix_library::operations::scale(a, 3);

    EXPECT_EQ(c.shape(), shape);
    EXPECT_EQ(c.size(), 4);
    EXPECT_EQ((c[{0, 0}]), 3);
    EXPECT_EQ((c[{0, 1}]), 6);
    EXPECT_EQ((c[{1, 0}]), 9);
    EXPECT_EQ((c[{1, 1}]), 12);
}

TEST(OperationsScaleTest, ScaleByZero)
{
    std::vector<std::size_t> shape = {2, 3};
    matrix_library::Tensor<int> a(shape);
    a.set_data(std::vector<int>{1, -1, 2, -2, 3, -3});

    auto c = matrix_library::operations::scale(a, 0);

    for (size_t i = 0; i < c.size(); ++i)
    {
        // Access via flat index conversion to multi-index is available through operator[] with vector
        // but for simplicity, reconstruct indices similarly to other tests by asking shape
    }

    EXPECT_EQ(c.size(), 6);
    for (size_t i = 0; i < c.size(); ++i)
    {
        // compute multi-index
        size_t remainder = i;
        const auto &shapeRef = c.shape();
        std::vector<size_t> idx(shapeRef.size());
        for (size_t dim = 0; dim < shapeRef.size(); ++dim)
        {
            size_t stride = 1;
            for (size_t k = dim + 1; k < shapeRef.size(); ++k)
                stride *= shapeRef[k];
            idx[dim] = remainder / stride;
            remainder = remainder % stride;
        }
        EXPECT_EQ(c[idx], 0);
    }
}

TEST(OperationsScaleTest, EmptyTensor)
{
    matrix_library::Tensor<double> a(std::vector<std::size_t>{});
    auto c = matrix_library::operations::scale(a, 5.0);
    EXPECT_EQ(c.size(), 0);
    EXPECT_EQ(c.shape(), std::vector<std::size_t>{});
}
