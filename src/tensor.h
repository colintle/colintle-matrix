#ifndef MATRIX_LIBRARY_TENSOR_H
#define MATRIX_LIBRARY_TENSOR_H

#include <cstddef>
#include <type_traits>
#include <vector>

namespace matrix_library
{
    template <class T>
    class Tensor
    {
    public:
        Tensor(const std::vector<std::size_t> &shape);

        T &operator[](const std::vector<std::size_t> &indices);
        const T &operator[](const std::vector<std::size_t> &indices) const;

        const std::vector<size_t> &shape() const;
        std::size_t size() const;

        void set_data(const std::vector<T> &data);
        void set_data(std::vector<T> &&data);
        void fill(const T &value);

    private:
        void generate_strides();
        size_t get_flat_index(const std::vector<std::size_t> &indices) const;

        std::vector<T> data_;
        std::vector<size_t> shape_;
        // A strides vector to help with index calculations
        // ``strides_[i]`` gives the number of elements to skip to move to the next index in dimension ``i``
        // For example, for a tensor with shape {2, 3}, strides would be {3, 1}
        // Using an indices vector {1, 2} would yield a flat index of 1*3 + 2*1 = 5
        // Skip 3 elements to move to the next row, and 1 element to move to the next column

        // Another example for a tensor with shape {2, 3, 4}:
        // Strides would be {12, 4, 1}
        // Skip 12 elements to move to the next 3-D layer, 4 elements to move to the next row, and 1 element to move to the next column
        // Using an indices vector {1, 2, 3} would yield a flat index of 1*12 + 2*4 + 3*1 = 23
        std::vector<size_t> strides_;
        std::size_t size_;
    };
};

#endif