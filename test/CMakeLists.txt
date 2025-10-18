add_executable(TensorTests tensor_test.cc)
target_link_libraries(TensorTests PRIVATE MatrixLibraryAPI gtest_main)

include(GoogleTest)
gtest_discover_tests(TensorTests)
