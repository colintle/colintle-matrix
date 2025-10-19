add_library(MatrixLibraryAPI tensor.cc)
target_include_directories(MatrixLibraryAPI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/operations)

add_subdirectory(operations)
target_link_libraries(MatrixLibraryAPI PUBLIC MatrixOperations)