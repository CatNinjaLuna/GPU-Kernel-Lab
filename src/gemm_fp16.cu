#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#define TILE_SIZE 16

// Naive FP16 GEMM kernel
__global__ void gemm_fp16_naive_kernel(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

// Tiled FP16 GEMM kernel with shared memory
__global__ void gemm_fp16_tiled_kernel(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    __shared__ __half As[TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = __float2half(sum);
    }
}

// Host wrapper functions
void gemm_fp16_naive_gpu(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_fp16_naive_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void gemm_fp16_tiled_gpu(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_fp16_tiled_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

// CPU reference implementation for FP16
void gemm_fp16_cpu(__half* A, __half* B, __half* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = __float2half(sum);
        }
    }
}
