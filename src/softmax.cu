#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Warp-level reduction for max
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Simple row-wise softmax kernel (each block processes one row)
__global__ void softmax_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];
    
    // Each block computes softmax for the entire vector
    if (blockIdx.x > 0) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Find max value for numerical stability
    float thread_max = -INFINITY;
    for (int i = tid; i < n; i += stride) {
        thread_max = fmaxf(thread_max, input[i]);
    }
    shared[tid] = thread_max;
    __syncthreads();
    
    // Reduction in shared memory to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        thread_sum += exp_val;
    }
    shared[tid] = thread_sum;
    __syncthreads();
    
    // Reduction to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float sum_val = shared[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < n; i += stride) {
        output[i] /= sum_val;
    }
}

void softmax_gpu(const float* input, float* output, int n) {
    const int blockSize = 256;
    size_t shared_mem_size = blockSize * sizeof(float);
    
    // Single block processes entire vector
    softmax_kernel<<<1, blockSize, shared_mem_size>>>(input, output, n);
    cudaDeviceSynchronize();
}

// CPU reference implementation
void softmax_cpu(const float* input, float* output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}
