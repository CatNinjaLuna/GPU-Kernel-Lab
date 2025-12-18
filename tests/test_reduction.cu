#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward declarations
extern void reduce_atomic_gpu(const float* input, float* output, int n);
extern void reduce_shared_gpu(const float* input, float* output, int n);
extern void reduce_warp_gpu(const float* input, float* output, int n);
extern float reduce_cpu(const float* input, int n);

bool verify_result(float gpu_result, float cpu_result, float epsilon = 1e-2) {
    float relative_error = fabs(gpu_result - cpu_result) / fabs(cpu_result);
    if (relative_error > epsilon) {
        printf("Mismatch: GPU=%f, CPU=%f, Error=%.6f%%\n", 
               gpu_result, cpu_result, relative_error * 100);
        return false;
    }
    return true;
}

void run_reduction_test(const char* variant_name, 
                       void (*reduce_func)(const float*, float*, int),
                       const float* d_input, float* d_output, 
                       const float* h_input, int n, float cpu_result) {
    printf("\n%s:\n", variant_name);
    
    // Warm-up
    reduce_func(d_input, d_output, n);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int num_runs = 10;
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; i++) {
        reduce_func(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= num_runs;
    
    // Copy result back
    float gpu_result;
    cudaMemcpy(&gpu_result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify and report
    bool correct = verify_result(gpu_result, cpu_result);
    float bandwidth_gbs = (n * sizeof(float)) / (time_ms * 1e6);
    
    printf("  Time: %.3f ms\n", time_ms);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbs);
    printf("  Result: %.2f (expected %.2f)\n", gpu_result, cpu_result);
    printf("  Status: %s\n", correct ? "PASSED" : "FAILED");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    int n = 10000000; // 10M elements
    
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Reduction Test (n=%d)\n", n);
    printf("================================\n");
    
    // Allocate host memory
    float *h_input = new float[n];
    
    // Initialize with known pattern for easy verification
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f; // Sum should be n
    }
    
    // CPU reference
    float cpu_result = reduce_cpu(h_input, n);
    printf("\nCPU Result: %.2f\n", cpu_result);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Test all variants
    run_reduction_test("Atomic (Naive)", reduce_atomic_gpu, 
                      d_input, d_output, h_input, n, cpu_result);
    
    run_reduction_test("Shared Memory", reduce_shared_gpu, 
                      d_input, d_output, h_input, n, cpu_result);
    
    run_reduction_test("Warp Shuffle (Optimized)", reduce_warp_gpu, 
                      d_input, d_output, h_input, n, cpu_result);
    
    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n================================\n");
    printf("Summary: Warp shuffle is typically 2-5× faster than naive atomic\n");
    printf("Key optimizations:\n");
    printf("  1. Atomic → Shared memory reduces global memory contention\n");
    printf("  2. Warp shuffle eliminates shared memory bank conflicts\n");
    printf("  3. Grid-stride loop improves occupancy\n");
    
    return 0;
}
