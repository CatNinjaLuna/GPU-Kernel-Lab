#include <cuda_runtime.h>
#include <cstdio>

#define TILE_WIDTH 16
#define MAX_KERNEL_SIZE 9

// Tiled 2D Convolution kernel with shared memory
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                               int height, int width, int ksize) {
    __shared__ float s_input[TILE_WIDTH + MAX_KERNEL_SIZE - 1][TILE_WIDTH + MAX_KERNEL_SIZE - 1];
    __shared__ float s_kernel[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int pad = ksize / 2;
    
    // Load kernel into shared memory (first threads)
    if (tx < ksize && ty < ksize) {
        s_kernel[ty][tx] = kernel[ty * ksize + tx];
    }
    
    // Load input tile into shared memory with halo
    int tile_size = TILE_WIDTH + ksize - 1;
    for (int i = ty; i < tile_size; i += blockDim.y) {
        for (int j = tx; j < tile_size; j += blockDim.x) {
            int input_row = blockIdx.y * TILE_WIDTH + i - pad;
            int input_col = blockIdx.x * TILE_WIDTH + j - pad;
            
            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                s_input[i][j] = input[input_row * width + input_col];
            } else {
                s_input[i][j] = 0.0f; // Padding
            }
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int ki = 0; ki < ksize; ki++) {
            for (int kj = 0; kj < ksize; kj++) {
                int si = ty + ki;
                int sj = tx + kj;
                sum += s_input[si][sj] * s_kernel[ki][kj];
            }
        }
        output[row * width + col] = sum;
    }
}

void conv2d_gpu(const float* input, const float* kernel, float* output,
                int height, int width, int ksize) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    
    conv2d_kernel<<<gridSize, blockSize>>>(input, kernel, output, height, width, ksize);
    cudaDeviceSynchronize();
}

// CPU reference implementation
void conv2d_cpu(const float* input, const float* kernel, float* output,
                int height, int width, int ksize) {
    int pad = ksize / 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < ksize; ki++) {
                for (int kj = 0; kj < ksize; kj++) {
                    int row = i + ki - pad;
                    int col = j + kj - pad;
                    if (row >= 0 && row < height && col >= 0 && col < width) {
                        sum += input[row * width + col] * kernel[ki * ksize + kj];
                    }
                }
            }
            output[i * width + j] = sum;
        }
    }
}
