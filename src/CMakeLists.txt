# CUDA kernel library
add_library(kernels STATIC
    vec_add.cu
    gemm.cu
    gemm_fp16.cu
    gemm_wmma.cu
    softmax.cu
    conv2d.cu
    reduction.cu
)

target_link_libraries(kernels CUDA::cudart)
