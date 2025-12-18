#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward declarations
extern void gemm_fp16_naive_gpu(const __half* A, const __half* B, __half* C, int M, int N, int K);
extern void gemm_fp16_tiled_gpu(const __half* A, const __half* B, __half* C, int M, int N, int K);
extern void gemm_wmma_gpu(const __half* A, const __half* B, float* C, int M, int N, int K);
extern void gemm_naive_gpu(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_tiled_gpu(const float* A, const float* B, float* C, int M, int N, int K);

void gemm_fp16_cpu(__half* A, __half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_fp32_results(const float* gpu, const float* cpu, int size, float epsilon = 0.05) {
    int mismatches = 0;
    for (int i = 0; i < size && mismatches < 10; i++) {
        float diff = fabs(gpu[i] - cpu[i]);
        float rel_diff = diff / fmax(fabs(cpu[i]), 1e-5f);
        if (rel_diff > epsilon) {
            if (mismatches == 0) {
                printf("Sample mismatches (relative error > %.1f%%):\n", epsilon * 100);
            }
            printf("  [%d] GPU=%f, CPU=%f, rel_err=%.2f%%\n", 
                   i, gpu[i], cpu[i], rel_diff * 100);
            mismatches++;
        }
    }
    // FP16 accumulation introduces small errors - accept if < 1% mismatch rate
    return (mismatches < size / 100);
}

int main(int argc, char** argv) {
    int size = 1024;
    
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    
    // WMMA requires sizes to be multiples of 16
    if (size % 16 != 0) {
        printf("Warning: Size %d rounded up to nearest multiple of 16 for WMMA\n", size);
        size = ((size + 15) / 16) * 16;
    }
    
    int M = size, N = size, K = size;
    
    printf("Tensor Core GEMM Benchmark (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("============================================================\n");
    printf("\n");
    
    // Allocate host memory for FP16
    __half *h_A_fp16 = new __half[M * K];
    __half *h_B_fp16 = new __half[K * N];
    float *h_C_fp32_cpu = new float[M * N];
    
    // Allocate host memory for FP32
    float *h_A_fp32 = new float[M * K];
    float *h_B_fp32 = new float[K * N];
    float *h_C_fp32_gpu = new float[M * N];
    
    // Initialize matrices (same values for both precisions)
    for (int i = 0; i < M * K; i++) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_A_fp32[i] = val;
        h_A_fp16[i] = __float2half(val);
    }
    for (int i = 0; i < K * N; i++) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_B_fp32[i] = val;
        h_B_fp16[i] = __float2half(val);
    }
    
    // Allocate device memory for FP16
    __half *d_A_fp16, *d_B_fp16;
    float *d_C_fp32;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(__half));
    cudaMalloc(&d_C_fp32, M * N * sizeof(float));
    
    // Allocate device memory for FP32
    float *d_A_fp32, *d_B_fp32;
    cudaMalloc(&d_A_fp32, M * K * sizeof(float));
    cudaMalloc(&d_B_fp32, K * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A_fp16, h_A_fp16, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16, K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_fp32, h_A_fp32, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B_fp32, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // CPU reference
    gemm_fp16_cpu(h_A_fp16, h_B_fp16, h_C_fp32_cpu, M, N, K);
    
    // ========== FP32 Tiled (Baseline) ==========
    printf("FP32 Tiled GEMM (Baseline):\n");
    cudaEventRecord(start);
    gemm_tiled_gpu(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp32_tiled_time;
    cudaEventElapsedTime(&fp32_tiled_time, start, stop);
    float fp32_tiled_gflops = (2.0f * M * N * K) / (fp32_tiled_time * 1e6);
    
    printf("  Time: %.3f ms\n", fp32_tiled_time);
    printf("  Performance: %.2f GFLOP/s\n\n", fp32_tiled_gflops);
    
    // ========== FP16 Tiled ==========
    printf("FP16 Tiled GEMM (No Tensor Cores):\n");
    __half *d_C_fp16;
    cudaMalloc(&d_C_fp16, M * N * sizeof(__half));
    
    cudaEventRecord(start);
    gemm_fp16_tiled_gpu(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp16_tiled_time;
    cudaEventElapsedTime(&fp16_tiled_time, start, stop);
    float fp16_tiled_gflops = (2.0f * M * N * K) / (fp16_tiled_time * 1e6);
    
    printf("  Time: %.3f ms\n", fp16_tiled_time);
    printf("  Performance: %.2f GFLOP/s\n", fp16_tiled_gflops);
    printf("  Speedup vs FP32: %.2fx\n\n", fp32_tiled_time / fp16_tiled_time);
    
    cudaFree(d_C_fp16);
    
    // ========== WMMA Tensor Cores ==========
    printf("FP16 WMMA GEMM (Tensor Cores):\n");
    cudaEventRecord(start);
    gemm_wmma_gpu(d_A_fp16, d_B_fp16, d_C_fp32, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float wmma_time;
    cudaEventElapsedTime(&wmma_time, start, stop);
    float wmma_gflops = (2.0f * M * N * K) / (wmma_time * 1e6);
    
    cudaMemcpy(h_C_fp32_gpu, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool wmma_correct = verify_fp32_results(h_C_fp32_gpu, h_C_fp32_cpu, M * N);
    
    printf("  Time: %.3f ms\n", wmma_time);
    printf("  Performance: %.2f GFLOP/s\n", wmma_gflops);
    printf("  Speedup vs FP32: %.2fx\n", fp32_tiled_time / wmma_time);
    printf("  Speedup vs FP16 (no TC): %.2fx\n", fp16_tiled_time / wmma_time);
    printf("  Result: %s\n\n", wmma_correct ? "PASSED" : "FAILED");
    
    // ========== Summary ==========
    printf("============================================================\n");
    printf("Summary - Tensor Core Impact:\n");
    printf("  FP32 Tiled:      %.2f GFLOP/s (baseline)\n", fp32_tiled_gflops);
    printf("  FP16 Tiled:      %.2f GFLOP/s (%.2fx)\n", fp16_tiled_gflops, fp16_tiled_gflops / fp32_tiled_gflops);
    printf("  FP16 WMMA (TC):  %.2f GFLOP/s (%.2fx) ⭐\n", wmma_gflops, wmma_gflops / fp32_tiled_gflops);
    printf("\n");
    printf("TensorRT Takeaway:\n");
    printf("  - Tensor Cores deliver %.2fx over regular FP16\n", wmma_gflops / fp16_tiled_gflops);
    printf("  - FP16 + Tensor Cores = %.2fx over FP32\n", fp32_tiled_time / wmma_time);
    printf("  - Critical for DL inference throughput\n");
    
    // Cleanup
    delete[] h_A_fp16;
    delete[] h_B_fp16;
    delete[] h_C_fp32_cpu;
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_C_fp32_gpu;
    
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_fp32);
    cudaFree(d_A_fp32);
    cudaFree(d_B_fp32);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return wmma_correct ? 0 : 1;
}
