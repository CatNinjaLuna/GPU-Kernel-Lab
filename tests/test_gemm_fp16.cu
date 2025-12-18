#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward declarations
extern void gemm_fp16_naive_gpu(const __half* A, const __half* B, __half* C, int M, int N, int K);
extern void gemm_fp16_tiled_gpu(const __half* A, const __half* B, __half* C, int M, int N, int K);
extern void gemm_fp16_cpu(__half* A, __half* B, __half* C, int M, int N, int K);
extern void gemm_naive_gpu(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_tiled_gpu(const float* A, const float* B, float* C, int M, int N, int K);

bool verify_fp16_results(const __half* gpu, const __half* cpu, int size, float epsilon = 1e-2) {
    for (int i = 0; i < size; i++) {
        float diff = fabs(__half2float(gpu[i]) - __half2float(cpu[i]));
        if (diff > epsilon) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f, diff=%f\n", 
                   i, __half2float(gpu[i]), __half2float(cpu[i]), diff);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int size = 1024;
    
    if (argc > 1) {
        size = atoi(argv[1]);
    }
    
    int M = size, N = size, K = size;
    
    printf("Mixed Precision GEMM Comparison (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("============================================================\n");
    printf("\n");
    
    // Allocate host memory for FP16
    __half *h_A_fp16 = new __half[M * K];
    __half *h_B_fp16 = new __half[K * N];
    __half *h_C_fp16_gpu = new __half[M * N];
    __half *h_C_fp16_cpu = new __half[M * N];
    
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
    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(__half));
    cudaMalloc(&d_C_fp16, M * N * sizeof(__half));
    
    // Allocate device memory for FP32
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    cudaMalloc(&d_A_fp32, M * K * sizeof(float));
    cudaMalloc(&d_B_fp32, K * N * sizeof(float));
    cudaMalloc(&d_C_fp32, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A_fp16, h_A_fp16, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16, K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_fp32, h_A_fp32, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32, h_B_fp32, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========== FP32 Naive ==========
    printf("FP32 Naive GEMM:\n");
    cudaEventRecord(start);
    gemm_naive_gpu(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp32_naive_time;
    cudaEventElapsedTime(&fp32_naive_time, start, stop);
    float fp32_naive_gflops = (2.0f * M * N * K) / (fp32_naive_time * 1e6);
    
    printf("  Time: %.3f ms\n", fp32_naive_time);
    printf("  Performance: %.2f GFLOP/s\n\n", fp32_naive_gflops);
    
    // ========== FP32 Tiled ==========
    printf("FP32 Tiled GEMM:\n");
    cudaEventRecord(start);
    gemm_tiled_gpu(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp32_tiled_time;
    cudaEventElapsedTime(&fp32_tiled_time, start, stop);
    float fp32_tiled_gflops = (2.0f * M * N * K) / (fp32_tiled_time * 1e6);
    
    printf("  Time: %.3f ms\n", fp32_tiled_time);
    printf("  Performance: %.2f GFLOP/s\n\n", fp32_tiled_gflops);
    
    // ========== FP16 Naive ==========
    printf("FP16 Naive GEMM:\n");
    cudaEventRecord(start);
    gemm_fp16_naive_gpu(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp16_naive_time;
    cudaEventElapsedTime(&fp16_naive_time, start, stop);
    float fp16_naive_gflops = (2.0f * M * N * K) / (fp16_naive_time * 1e6);
    
    cudaMemcpy(h_C_fp16_gpu, d_C_fp16, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    gemm_fp16_cpu(h_A_fp16, h_B_fp16, h_C_fp16_cpu, M, N, K);
    bool fp16_naive_correct = verify_fp16_results(h_C_fp16_gpu, h_C_fp16_cpu, M * N);
    
    printf("  Time: %.3f ms\n", fp16_naive_time);
    printf("  Performance: %.2f GFLOP/s\n", fp16_naive_gflops);
    printf("  Speedup vs FP32 Naive: %.2fx\n", fp32_naive_time / fp16_naive_time);
    printf("  Result: %s\n\n", fp16_naive_correct ? "PASSED" : "FAILED");
    
    // ========== FP16 Tiled ==========
    printf("FP16 Tiled GEMM:\n");
    cudaEventRecord(start);
    gemm_fp16_tiled_gpu(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fp16_tiled_time;
    cudaEventElapsedTime(&fp16_tiled_time, start, stop);
    float fp16_tiled_gflops = (2.0f * M * N * K) / (fp16_tiled_time * 1e6);
    
    cudaMemcpy(h_C_fp16_gpu, d_C_fp16, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    bool fp16_tiled_correct = verify_fp16_results(h_C_fp16_gpu, h_C_fp16_cpu, M * N);
    
    printf("  Time: %.3f ms\n", fp16_tiled_time);
    printf("  Performance: %.2f GFLOP/s\n", fp16_tiled_gflops);
    printf("  Speedup vs FP32 Tiled: %.2fx\n", fp32_tiled_time / fp16_tiled_time);
    printf("  Result: %s\n\n", fp16_tiled_correct ? "PASSED" : "FAILED");
    
    // ========== Summary ==========
    printf("============================================================\n");
    printf("Summary:\n");
    printf("  FP16 Memory Footprint: %.2f%% of FP32\n", 50.0f);
    printf("  FP16 Bandwidth Savings: 2x (16-bit vs 32-bit)\n");
    printf("  Best FP16 Speedup: %.2fx (Tiled)\n", fp32_tiled_time / fp16_tiled_time);
    printf("  TensorRT Benefit: Lower latency + memory usage\n");
    
    // Cleanup
    delete[] h_A_fp16;
    delete[] h_B_fp16;
    delete[] h_C_fp16_gpu;
    delete[] h_C_fp16_cpu;
    delete[] h_A_fp32;
    delete[] h_B_fp32;
    delete[] h_C_fp32_gpu;
    
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_fp16);
    cudaFree(d_A_fp32);
    cudaFree(d_B_fp32);
    cudaFree(d_C_fp32);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (fp16_naive_correct && fp16_tiled_correct) ? 0 : 1;
}
