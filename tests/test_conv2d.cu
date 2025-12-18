#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Forward declarations
extern void conv2d_gpu(const float* input, const float* kernel, float* output,
                       int height, int width, int ksize);
extern void conv2d_cpu(const float* input, const float* kernel, float* output,
                       int height, int width, int ksize);

bool verify_results(const float* gpu, const float* cpu, int size, float epsilon = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int height = 512, width = 512, ksize = 3;
    
    printf("Conv2D Test (%dx%d, kernel=%dx%d)\n", height, width, ksize, ksize);
    printf("================================\n");
    
    int input_size = height * width;
    int kernel_size = ksize * ksize;
    
    // Allocate and initialize
    float *h_input = new float[input_size];
    float *h_kernel = new float[kernel_size];
    float *h_output_gpu = new float[input_size];
    float *h_output_cpu = new float[input_size];
    
    for (int i = 0; i < input_size; i++) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernel_size; i++) h_kernel[i] = 1.0f / kernel_size; // Average filter
    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU version
    conv2d_gpu(d_input, d_kernel, d_output, height, width, ksize);
    cudaMemcpy(h_output_gpu, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Run CPU version
    conv2d_cpu(h_input, h_kernel, h_output_cpu, height, width, ksize);
    
    // Verify
    bool correct = verify_results(h_output_gpu, h_output_cpu, input_size);
    printf("Result: %s\n", correct ? "PASSED" : "FAILED (TODO: Implement conv2d kernel)");
    
    // Cleanup
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output_gpu;
    delete[] h_output_cpu;
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return 0;
}
