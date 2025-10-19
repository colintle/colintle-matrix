#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"
#include "operations/multiply.h"

TEST(OperationsMultiplyTest, SimpleMultiply2D)
{
	std::vector<std::size_t> a_shape = {2, 2};
	std::vector<std::size_t> b_shape = {2, 2};
	matrix_library::Tensor<int> a(a_shape);
	matrix_library::Tensor<int> b(b_shape);

	a.set_data(std::vector<int>{1, 2, 3, 4});
	b.set_data(std::vector<int>{10, 20, 30, 40});

	auto c = matrix_library::operations::multiply(a, b);

	EXPECT_EQ(c.shape(), (std::vector<std::size_t>{2, 2}));
	EXPECT_EQ(c.size(), 4);
	EXPECT_EQ((c[{0, 0}]), 1 * 10 + 2 * 30);
	EXPECT_EQ((c[{0, 1}]), 1 * 20 + 2 * 40);
	EXPECT_EQ((c[{1, 0}]), 3 * 10 + 4 * 30);
	EXPECT_EQ((c[{1, 1}]), 3 * 20 + 4 * 40);
}

TEST(OperationsMultiplyTest, ShapeMismatchThrows)
{
	matrix_library::Tensor<int> a(std::vector<std::size_t>{2, 3});
	matrix_library::Tensor<int> b(std::vector<std::size_t>{4, 2});

	EXPECT_THROW(matrix_library::operations::multiply(a, b), std::invalid_argument);
}

TEST(OperationsMultiplyTest, Non2DThrows)
{
	matrix_library::Tensor<int> a(std::vector<std::size_t>{});
	matrix_library::Tensor<int> b(std::vector<std::size_t>{});

	EXPECT_THROW(matrix_library::operations::multiply(a, b), std::invalid_argument);
}

TEST(OperationsMultiplyTest, VectorDot1D)
{
	matrix_library::Tensor<int> a(std::vector<std::size_t>{3});
	matrix_library::Tensor<int> b(std::vector<std::size_t>{3});
	a.set_data(std::vector<int>{1, 2, 3});
	b.set_data(std::vector<int>{4, 5, 6});

	auto c = matrix_library::operations::multiply(a, b);
	// Result is 1x1 tensor containing dot product: 1*4 + 2*5 + 3*6 = 32
	EXPECT_EQ(c.shape(), (std::vector<std::size_t>{1, 1}));
	EXPECT_EQ(c.size(), 1);
	EXPECT_EQ((c[{0, 0}]), 32);
}

TEST(OperationsMultiplyTest, VectorTimesMatrix1D2D)
{
	matrix_library::Tensor<int> v(std::vector<std::size_t>{2});
	matrix_library::Tensor<int> m(std::vector<std::size_t>{2, 2});
	v.set_data(std::vector<int>{3, 4});
	m.set_data(std::vector<int>{1, 2, 3, 4}); // [[1,2],[3,4]]

	auto r = matrix_library::operations::multiply(v, m);
	// v treated as 1x2 row vector: result is 1x2: [3*1+4*3, 3*2+4*4] = [15, 22]
	EXPECT_EQ(r.shape(), (std::vector<std::size_t>{1, 2}));
	EXPECT_EQ((r[{0, 0}]), 15);
	EXPECT_EQ((r[{0, 1}]), 22);
}

TEST(OperationsMultiplyTest, MatrixTimesVector2D1D)
{
	matrix_library::Tensor<int> m(std::vector<std::size_t>{2, 2});
	matrix_library::Tensor<int> v(std::vector<std::size_t>{2});
	m.set_data(std::vector<int>{1, 2, 3, 4});
	v.set_data(std::vector<int>{5, 6});

	auto r = matrix_library::operations::multiply(m, v);
	// result is 2x1: [1*5+2*6, 3*5+4*6] = [17, 39]
	EXPECT_EQ(r.shape(), (std::vector<std::size_t>{2, 1}));
	EXPECT_EQ((r[{0, 0}]), 17);
	EXPECT_EQ((r[{1, 0}]), 39);
}
