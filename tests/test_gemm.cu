#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward declarations
extern void gemm_naive_gpu(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_tiled_gpu(const float* A, const float* B, float* C, int M, int N, int K);
extern void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

bool verify_results(const float* gpu, const float* cpu, int size, float epsilon = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, gpu[i], cpu[i]);
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
    
    printf("GEMM Test (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("================================\n");
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C_gpu = new float[M * N];
    float *h_C_cpu = new float[M * N];
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test tiled version
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    gemm_tiled_gpu(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Run CPU version
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
    
    // Verify results
    bool correct = verify_results(h_C_gpu, h_C_cpu, M * N);
    
    float gflops = (2.0f * M * N * K) / (gpu_time * 1e6);
    
    printf("GPU Time: %.3f ms\n", gpu_time);
    printf("Performance: %.2f GFLOP/s\n", gflops);
    printf("Result: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return correct ? 0 : 1;
}
