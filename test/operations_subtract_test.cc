#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"
#include "operations/subtract.h"

TEST(OperationsSubtractTest, SimpleSubtract2D)
{
	std::vector<std::size_t> shape = {2, 2};
	matrix_library::Tensor<int> a(shape);
	matrix_library::Tensor<int> b(shape);

	a.set_data(std::vector<int>{10, 20, 30, 40});
	b.set_data(std::vector<int>{1, 2, 3, 4});

	auto c = matrix_library::operations::subtract(a, b);

	EXPECT_EQ(c.shape(), shape);
	EXPECT_EQ(c.size(), 4);
	EXPECT_EQ((c[{0, 0}]), 9);
	EXPECT_EQ((c[{0, 1}]), 18);
	EXPECT_EQ((c[{1, 0}]), 27);
	EXPECT_EQ((c[{1, 1}]), 36);
}

TEST(OperationsSubtractTest, ShapeMismatchThrows)
{
	matrix_library::Tensor<int> a(std::vector<std::size_t>{2, 2});
	matrix_library::Tensor<int> b(std::vector<std::size_t>{3});

	EXPECT_THROW(matrix_library::operations::subtract(a, b), std::invalid_argument);
}

TEST(OperationsSubtractTest, EmptyTensors)
{
	matrix_library::Tensor<int> a(std::vector<std::size_t>{});
	matrix_library::Tensor<int> b(std::vector<std::size_t>{});

	auto c = matrix_library::operations::subtract(a, b);
	EXPECT_EQ(c.size(), 0);
	EXPECT_EQ(c.shape(), std::vector<std::size_t>{});
}