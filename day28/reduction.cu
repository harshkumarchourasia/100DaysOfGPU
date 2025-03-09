#include "solve.h"
#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void sum_reduction(const float *input, float *output, int N)
{
    __shared__ float input_s[BLOCK_DIM];
    int segment = 2 * blockDim.x * blockIdx.x;
    int i = segment + threadIdx.x;
    int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            input_s[t] += input_s[t + stride];
    }
    if (t == 0)
        atomicAdd(output, input_s[0]);
}

// input, output are device pointers
void solve(const float *input, float *output, int N)
{
    sum_reduction<<<(N + 1023) / 1024, BLOCK_DIM>>>(input, output, N);
}