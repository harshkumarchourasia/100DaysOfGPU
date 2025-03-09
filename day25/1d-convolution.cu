#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int i = bx * blockDim.x + tx;

    // Use externally allocated shared memory
    extern __shared__ float input_s[];  // Now this array gets its size from kernel launch

    if (i < input_size) input_s[tx] = input[i];
    else input_s[tx] = 0.0f;

    if (tx < kernel_size) {
        if (i + 256 < input_size)
            input_s[tx + 256] = input[i + 256];
        else
            input_s[tx + 256] = 0.0f;
    }

    __syncthreads();

    float value = 0;
    for (int iter = 0; iter < kernel_size; iter++) {
        value += input_s[tx + iter] * kernel[iter];  // Use shared memory, not global memory
    }

    if (i < input_size - kernel_size + 1)
        output[i] = value;
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    int sharedMemSize = (threadsPerBlock + kernel_size) * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}