#ifndef MATRIX_LIBRARY_OPERATIONS_TRANSPOSE_H
#define MATRIX_LIBRARY_OPERATIONS_TRANSPOSE_H

#include "../tensor.h"
#include <stdexcept>

namespace matrix_library::operations
{
    template <class T>
    matrix_library::Tensor<T> transpose(const matrix_library::Tensor<T> &a)
    {
        const auto &shape = a.shape();

        if (shape.size() == 0)
        {
            // Transpose of a scalar/empty-shape: return same
            return matrix_library::Tensor<T>(std::vector<std::size_t>{});
        }

        if (shape.size() == 1)
        {
            // Treat 1D vector as column vector -> return 2D with shape {N,1}
            std::vector<std::size_t> outShape = {shape[0], 1};
            matrix_library::Tensor<T> out(outShape);
            for (size_t i = 0; i < shape[0]; ++i)
                out[{i, 0}] = a[{i}];
            return out;
        }

        if (shape.size() == 2)
        {
            std::vector<std::size_t> outShape = {shape[1], shape[0]};
            matrix_library::Tensor<T> out(outShape);
            for (size_t i = 0; i < shape[0]; ++i)
            {
                for (size_t j = 0; j < shape[1]; ++j)
                {
                    out[{j, i}] = a[{i, j}];
                }
            }
            return out;
        }

        throw std::invalid_argument("Transpose currently supports only rank 0-2 tensors.");
    }
}

#endif
