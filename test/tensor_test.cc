#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"

TEST(TensorTest, ConstructorInitializesShapeAndSize) {
    std::vector<std::size_t> shape = {2, 3, 4};
    matrix_library::Tensor<int> tensor(shape);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ((tensor[{1, 2, 3}]), 0);
}
