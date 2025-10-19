#ifndef MATRIX_LIBRARY_OPERATIONS_MULTIPLY_H
#define MATRIX_LIBRARY_OPERATIONS_MULTIPLY_H

#include "../tensor.h"
#include <stdexcept>

namespace matrix_library::operations
{
	template <class T>
	matrix_library::Tensor<T> multiply(const matrix_library::Tensor<T> &a, const matrix_library::Tensor<T> &b)
	{
		const std::vector<size_t> &ashape = a.shape();
		const std::vector<size_t> &bshape = b.shape();

		const size_t arank = ashape.size();
		const size_t brank = bshape.size();

		// Handle 1D x 1D -> return 1x1 matrix containing inner product
		if (arank == 1 && brank == 1)
		{
			size_t k = ashape[0];
			if (bshape[0] != k)
				throw std::invalid_argument("Inner dimensions must match for vector dot product.");

			matrix_library::Tensor<T> result(std::vector<std::size_t>{1, 1});
			T acc = T();
			for (size_t i = 0; i < k; ++i)
				acc += a[{i}] * b[{i}];
			result[{0, 0}] = acc;
			return result;
		}

		// Handle 1D x 2D: treat 1D as row vector (1 x k) * (k x n) -> (1 x n)
		if (arank == 1 && brank == 2)
		{
			size_t k = ashape[0];
			if (bshape[0] != k)
				throw std::invalid_argument("Inner dimensions must match for vector-matrix multiplication.");

			std::vector<std::size_t> resultShape = {1, bshape[1]};
			matrix_library::Tensor<T> result(resultShape);
			for (size_t j = 0; j < bshape[1]; ++j)
			{
				T acc = T();
				for (size_t kk = 0; kk < k; ++kk)
					acc += a[{kk}] * b[{kk, j}];
				result[{0, j}] = acc;
			}
			return result;
		}

		// Handle 2D x 1D: (m x k) * (k) treated as (m x k) * (k x 1) -> (m x 1)
		if (arank == 2 && brank == 1)
		{
			size_t m = ashape[0];
			size_t k = ashape[1];
			if (bshape[0] != k)
				throw std::invalid_argument("Inner dimensions must match for matrix-vector multiplication.");

			matrix_library::Tensor<T> result(std::vector<std::size_t>{m, 1});
			for (size_t i = 0; i < m; ++i)
			{
				T acc = T();
				for (size_t kk = 0; kk < k; ++kk)
					acc += a[{i, kk}] * b[{kk}];
				result[{i, 0}] = acc;
			}
			return result;
		}

		// Handle 2D x 2D
		if (arank == 2 && brank == 2)
		{
			if (ashape[1] != bshape[0])
				throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");

			std::vector<std::size_t> resultShape = {ashape[0], bshape[1]};
			matrix_library::Tensor<T> result(resultShape);
			for (size_t i = 0; i < ashape[0]; ++i)
			{
				for (size_t j = 0; j < bshape[1]; ++j)
				{
					T acc = T();
					for (size_t kk = 0; kk < ashape[1]; ++kk)
						acc += a[{i, kk}] * b[{kk, j}];
					result[{i, j}] = acc;
				}
			}
			return result;
		}

		throw std::invalid_argument("Unsupported tensor ranks for multiply (supported: 1D and 2D).");
	}
}

#endif
