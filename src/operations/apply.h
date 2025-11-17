#ifndef MATRIX_LIBRARY_OPERATIONS_APPLY_H
#define MATRIX_LIBRARY_OPERATIONS_APPLY_H

#include "../tensor.h"
#include <vector>

namespace matrix_library::operations {
// Apply a unary function to every element of a tensor in place.
// Uses brace indexing a[{...}] for rank 1-3 tensors, and a generic fallback
// otherwise.
template <class T, class F>
void apply_inplace(matrix_library::Tensor<T> &a, F &&func) {
  const auto &s = a.shape();

  std::vector<std::size_t> idx(s.size());
  for (size_t i = 0; i < a.size(); ++i) {
    size_t remainder = i;
    for (size_t dim = 0; dim < s.size(); ++dim) {
      size_t stride = 1;
      for (size_t k = dim + 1; k < s.size(); ++k)
        stride *= s[k];
      idx[dim] = remainder / stride;
      remainder = remainder % stride;
    }
    a[idx] = static_cast<T>(func(a[idx]));
  }
}

} // namespace matrix_library::operations

#endif
