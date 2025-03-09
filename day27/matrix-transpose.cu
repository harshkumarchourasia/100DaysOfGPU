#include "solve.h"
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float *input, float *output, int rows, int cols)
{
    int row = threadIdx.y + blockIdx.y * blockIdx.y;
    int col = threadIdx.x + blockIdx.x * blockIdx.x;
    int new_row = col;
    int new_col = row;
    if (row < rows && col < cols)
        output[rows * new_row + new_col] = input[cols * row + col];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float *input, float *output, int rows, int cols)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}