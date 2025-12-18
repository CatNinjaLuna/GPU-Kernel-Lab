#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward declarations
extern void softmax_gpu(const float* input, float* output, int n);
extern void softmax_cpu(const float* input, float* output, int n);

bool verify_results(const float* gpu, const float* cpu, int n, float epsilon = 1e-3) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int n = 1000;
    
    printf("Softmax Test (n=%d)\n", n);
    printf("================================\n");
    
    // Allocate and initialize
    float *h_input = new float[n];
    float *h_output_gpu = new float[n];
    float *h_output_cpu = new float[n];
    
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU version
    softmax_gpu(d_input, d_output, n);
    cudaMemcpy(h_output_gpu, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Run CPU version
    softmax_cpu(h_input, h_output_cpu, n);
    
    // Verify
    bool correct = verify_results(h_output_gpu, h_output_cpu, n);
    printf("Result: %s\n", correct ? "PASSED" : "FAILED (TODO: Implement softmax kernel)");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
