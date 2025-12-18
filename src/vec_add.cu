#include <cuda_runtime.h>
#include <cstdio>

// CUDA kernel for vector addition
__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Host wrapper function
void vec_add_gpu(const float* a, const float* b, float* c, int n) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    
    vec_add_kernel<<<numBlocks, blockSize>>>(a, b, c, n);
    cudaDeviceSynchronize();
}

// CPU reference implementation
void vec_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
