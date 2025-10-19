#ifndef MATRIX_LIBRARY_OPERATIONS_INVERSE_H
#define MATRIX_LIBRARY_OPERATIONS_INVERSE_H

#include "../tensor.h"
#include <stdexcept>
#include <vector>
#include <cmath>

namespace matrix_library::operations
{
    // Invert a square 2D matrix using Gauss-Jordan elimination.
    // Returns a Tensor<double> containing the inverse. Throws if not square or singular.
    template <class T>
    matrix_library::Tensor<double> inverse(const matrix_library::Tensor<T> &a)
    {
        const auto &shape = a.shape();
        if (shape.size() != 2)
            throw std::invalid_argument("Inverse requires a 2D square tensor.");
        if (shape[0] != shape[1])
            throw std::invalid_argument("Inverse requires a square matrix.");

        size_t n = shape[0];
        matrix_library::Tensor<double> mat(std::vector<size_t>{n, n});
        matrix_library::Tensor<double> inv(std::vector<size_t>{n, n});

        // Copy into double matrix and initialize inv to identity
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                mat[{i, j}] = static_cast<double>(a[{i, j}]);
                inv[{i, j}] = (i == j) ? 1.0 : 0.0;
            }
        }

        const double EPS = 1e-12;
        // Gauss-Jordan elimination
        for (size_t col = 0; col < n; ++col)
        {
            // Find pivot in the current column
            size_t pivot = col;
            double maxVal = std::fabs(mat[{pivot, col}]);
            for (size_t r = col + 1; r < n; ++r)
            {
                double val = std::fabs(mat[{r, col}]);
                if (val > maxVal)
                {
                    maxVal = val;
                    pivot = r;
                }
            }
            
            // if the max value is 0
            if (maxVal < EPS)
                throw std::invalid_argument("Matrix is singular and cannot be inverted.");

            // Swap rows if needed between the new pivot and old pivot
            if (pivot != col)
            {
                for (size_t c = 0; c < n; ++c)
                {
                    std::swap(mat[{col, c}], mat[{pivot, c}]);
                    std::swap(inv[{col, c}], inv[{pivot, c}]);
                }
            }

            // Normalize pivot row
            double diag = mat[{col, col}];
            for (size_t c = 0; c < n; ++c)
            {
                mat[{col, c}] /= diag;
                inv[{col, c}] /= diag;
            }

            // Eliminate other rows
            for (size_t r = 0; r < n; ++r)
            {
                if (r == col) continue;
                // This needs to be zeroed out at (r, col)
                double factor = mat[{r, col}];
                for (size_t c = 0; c < n; ++c)
                {
                    // {col, c} is technically 1
                    mat[{r, c}] -= factor * mat[{col, c}];
                    inv[{r, c}] -= factor * inv[{col, c}];
                }
            }
        }

        return inv;
    }
}

#endif
