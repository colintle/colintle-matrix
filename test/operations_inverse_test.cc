#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"
#include "operations/inverse.h"

TEST(OperationsInverseTest, SimpleInverse2x2)
{
    matrix_library::Tensor<double> a(std::vector<std::size_t>{2, 2});
    a.set_data(std::vector<double>{4, 7, 2, 6}); // [[4,7],[2,6]]

    auto inv = matrix_library::operations::inverse(a);

    // Known inverse: 1/(4*6-7*2) * [[6,-7],[-2,4]] => det = 10
    EXPECT_NEAR((inv[{0, 0}]), 6.0 / 10.0, 1e-9);
    EXPECT_NEAR((inv[{0, 1}]), -7.0 / 10.0, 1e-9);
    EXPECT_NEAR((inv[{1, 0}]), -2.0 / 10.0, 1e-9);
    EXPECT_NEAR((inv[{1, 1}]), 4.0 / 10.0, 1e-9);
}

TEST(OperationsInverseTest, SingularThrows)
{
    matrix_library::Tensor<double> a(std::vector<std::size_t>{2, 2});
    a.set_data(std::vector<double>{1, 2, 2, 4}); // rows are linearly dependent

    EXPECT_THROW(matrix_library::operations::inverse(a), std::invalid_argument);
}

TEST(OperationsInverseTest, NonSquareThrows)
{
    matrix_library::Tensor<double> a(std::vector<std::size_t>{2, 3});
    EXPECT_THROW(matrix_library::operations::inverse(a), std::invalid_argument);
}
