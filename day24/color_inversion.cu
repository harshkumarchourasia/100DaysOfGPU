#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char *image, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
    for (int channel = 0; channel < 3; channel++)
    {
        int idx = 4 * (row * width + col) + channel;
        if (idx >= width * height * 4)
            return;
        image[idx] = 255 - image[idx];
    }
}

void solve(unsigned char *image, int width, int height)
{
    unsigned char *d_image;
    int image_size = width * height * 4;

    // Allocate device memory
    cudaMalloc(&d_image, image_size * sizeof(unsigned char));

    // Copy input data from host to device
    cudaMemcpy(d_image, image, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(image, d_image, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
}