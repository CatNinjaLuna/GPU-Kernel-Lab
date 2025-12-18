#include <cuda_runtime.h>
#include <cstdio>

// Naive reduction using global memory atomics
__global__ void reduce_atomic_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        atomicAdd(output, input[i]);
    }
}

// Reduction with shared memory - tree-based
__global__ void reduce_shared_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Load data into shared memory with grid-stride loop
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    shared[tid] = sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // First thread writes block result to global memory
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// Warp-level reduction using shuffle intrinsics
__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized reduction with warp shuffle
__global__ void reduce_warp_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Grid-stride loop to accumulate values
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }
    
    // Warp-level reduction
    sum = warp_reduce(sum);
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction of warp results
    if (warp_id == 0) {
        sum = (lane_id < blockDim.x / 32) ? shared[lane_id] : 0.0f;
        sum = warp_reduce(sum);
        
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host wrapper functions
void reduce_atomic_gpu(const float* input, float* output, int n) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    
    cudaMemset(output, 0, sizeof(float));
    reduce_atomic_kernel<<<numBlocks, blockSize>>>(input, output, n);
    cudaDeviceSynchronize();
}

void reduce_shared_gpu(const float* input, float* output, int n) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    size_t shared_mem_size = blockSize * sizeof(float);
    
    cudaMemset(output, 0, sizeof(float));
    reduce_shared_kernel<<<numBlocks, blockSize, shared_mem_size>>>(input, output, n);
    cudaDeviceSynchronize();
}

void reduce_warp_gpu(const float* input, float* output, int n) {
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    size_t shared_mem_size = (blockSize / 32) * sizeof(float);
    
    cudaMemset(output, 0, sizeof(float));
    reduce_warp_kernel<<<numBlocks, blockSize, shared_mem_size>>>(input, output, n);
    cudaDeviceSynchronize();
}

// CPU reference implementation
float reduce_cpu(const float* input, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}
