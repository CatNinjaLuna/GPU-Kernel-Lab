# Test executables
add_executable(test_vec_add test_vec_add.cu)
target_link_libraries(test_vec_add kernels CUDA::cudart)

add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm kernels CUDA::cudart)

add_executable(test_softmax test_softmax.cu)
target_link_libraries(test_softmax kernels CUDA::cudart)

add_executable(test_conv2d test_conv2d.cu)
target_link_libraries(test_conv2d kernels CUDA::cudart)

add_executable(test_reduction test_reduction.cu)
target_link_libraries(test_reduction kernels CUDA::cudart)
