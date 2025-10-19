#ifndef MATRIX_LIBRARY_OPERATIONS_ADD_H
#define MATRIX_LIBRARY_OPERATIONS_ADD_H

#include "../tensor.h"
#include <stdexcept>

namespace matrix_library::operations
{
    template <class T>
    matrix_library::Tensor<T> add(const matrix_library::Tensor<T> &a, const matrix_library::Tensor<T> &b)
    {
        if (a.shape() != b.shape())
        {
            throw std::invalid_argument("Tensor shapes must match for addition.");
        }

        matrix_library::Tensor<T> result(a.shape());
        if (result.size() == 0)
        {
            return result;
        }

        std::vector<std::size_t> idx(a.shape().size());
        // Looping through each element (size())
        for (size_t i = 0; i < result.size(); ++i)
        {
            size_t remainder = i;
            const std::vector<size_t> &shape = a.shape();
            // Looping through each dimension
            for (size_t dim = 0; dim < shape.size(); ++dim)
            {
                size_t stride = 1;
                //
                for (size_t k = dim + 1; k < shape.size(); ++k)
                    stride *= shape[k];
                idx[dim] = remainder / stride;
                remainder = remainder % stride;
            }
            result[idx] = a[idx] + b[idx];
        }

        return result;
    }
}

#endif
