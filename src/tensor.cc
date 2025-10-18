#include "tensor.h"

namespace matrix_library
{
    template <class T>
    Tensor<T>::Tensor(const std::vector<std::size_t> &shape)
        : shape_(shape)
    {
        if (shape_.empty())
        {
            size_ = 0;
            return;
        }
        size_ = 1;
        for (std::size_t dim : shape_)
        {
            size_ *= dim;
        }
        data_.resize(size_);
        generate_strides();
    }

    template <class T>
    const std::vector<size_t> &Tensor<T>::shape() const
    {
        return shape_;
    }

    template <class T>
    std::size_t Tensor<T>::size() const
    {
        return size_;
    }

    template <class T>
    void Tensor<T>::generate_strides()
    {
        strides_.resize(shape_.size());
        if (shape_.empty())
        {
            return;
        }

        strides_.back() = 1;
        for (int i = shape_.size() - 2; i >= 0; --i)
        {
            // In a 3-d layer, when i is 0, strides_[i + 1] is the number of columns and shape_[i + 1] is the number of rows
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    template <class T>
    size_t Tensor<T>::get_flat_index(const std::vector<std::size_t> &indices) const
    {
        if (indices.size() != shape_.size())
        {
            throw std::out_of_range("Number of indices does not match tensor dimensions.");
        }

        size_t flat_index = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= shape_[i])
            {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
            flat_index += indices[i] * strides_[i];
        }
        return flat_index;
    }

    template <class T>
    T &Tensor<T>::operator[](const std::vector<std::size_t> &indices)
    {
        size_t flat_index = get_flat_index(indices);
        return data_[flat_index];
    }
    template <class T>
    const T &Tensor<T>::operator[](const std::vector<std::size_t> &indices) const
    {
        size_t flat_index = get_flat_index(indices);
        return data_[flat_index];
    }

    template <class T>
    void Tensor<T>::set_data(const std::vector<T> &data)
    {
        if (data.size() != size_)
        {
            throw std::invalid_argument("Input data size does not match tensor size.");
        }
        data_ = data;
    }

    template <class T>
    void Tensor<T>::set_data(std::vector<T> &&data)
    {
        if (data.size() != size_)
        {
            throw std::invalid_argument("Input data size does not match tensor size.");
        }
        data_ = std::move(data);
    }

    template <class T>
    void Tensor<T>::fill(const T &value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }

    template class Tensor<int>;
    template class Tensor<float>;
    template class Tensor<double>;
};