#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <type_traits>
#include <vector>

namespace MatrixLibrary
{
    template <class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
    std::vector<std::vector<T>> add(const std::vector<std::vector<T>> &A,
                                    const std::vector<std::vector<T>> &B);

    template <class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
    std::vector<std::vector<T>> subtract(const std::vector<std::vector<T>> &A,
                                         const std::vector<std::vector<T>> &B);

    template <class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
    std::vector<std::vector<T>> scale(const std::vector<std::vector<T>> &A, T scalar);

    template <class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
    std::vector<std::vector<T>> multiply(const std::vector<std::vector<T>> &A,
                                         const std::vector<std::vector<T>> &B);

    template <class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
    std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &A);

} 
#endif
