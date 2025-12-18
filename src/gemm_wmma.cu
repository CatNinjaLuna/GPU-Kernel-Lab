#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// WMMA Tensor Core kernel for FP16 GEMM
// Uses 16x16x16 matrix multiply-accumulate operations
__global__ void gemm_wmma_kernel(const __half* A, const __half* B, float* C, int M, int N, int K) {
    // Each warp computes one 16x16 output tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Calculate output position for this warp
    int rowStart = warpM * WMMA_M;
    int colStart = warpN * WMMA_N;
    
    // Bounds check for entire tile
    if (rowStart >= M || colStart >= N) {
        return;
    }
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Loop over K dimension in WMMA_K chunks
    for (int k = 0; k < K; k += WMMA_K) {
        // Load the inputs
        wmma::load_matrix_sync(a_frag, A + rowStart * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + colStart, N);
        
        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    // Store the output
    wmma::store_matrix_sync(C + rowStart * N + colStart, acc_frag, N, wmma::mem_row_major);
}

// Host wrapper for WMMA kernel
void gemm_wmma_gpu(const __half* A, const __half* B, float* C, int M, int N, int K) {
    // Launch one warp (32 threads) per 16x16 output tile
    // Use blocks of 128 threads (4 warps)
    dim3 blockSize(32, 4);  // 4 warps per block
    dim3 gridSize(
        (M + WMMA_M - 1) / WMMA_M,
        (N + WMMA_N - 1) / WMMA_N
    );
    
    gemm_wmma_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
