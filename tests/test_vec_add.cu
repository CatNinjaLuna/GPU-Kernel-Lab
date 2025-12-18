#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward declarations
extern void vec_add_gpu(const float* a, const float* b, float* c, int n);
extern void vec_add_cpu(const float* a, const float* b, float* c, int n);

bool verify_results(const float* gpu, const float* cpu, int n, float epsilon = 1e-5) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int n = 10000000; // 10M elements
    
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Vector Addition Test (n=%d)\n", n);
    printf("================================\n");
    
    // Allocate host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c_gpu = new float[n];
    float *h_c_cpu = new float[n];
    
    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    vec_add_gpu(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy result back
    cudaMemcpy(h_c_gpu, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Run CPU version
    vec_add_cpu(h_a, h_b, h_c_cpu, n);
    
    // Verify results
    bool correct = verify_results(h_c_gpu, h_c_cpu, n);
    
    printf("GPU Time: %.3f ms\n", gpu_time);
    printf("Bandwidth: %.2f GB/s\n", (3.0f * n * sizeof(float)) / (gpu_time * 1e6));
    printf("Result: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_gpu;
    delete[] h_c_cpu;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return correct ? 0 : 1;
}
