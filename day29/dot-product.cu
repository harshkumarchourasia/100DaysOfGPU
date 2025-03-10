#include "solve.h"
#include <cuda_runtime.h>
#define BLOCK_DIM 1024

__global__ void dot_product(const float *A, const float *B, float *result, int N)
{
    int segment = 2 * blockDim.x * blockIdx.x;
    int t = threadIdx.x;
    int i = segment + t;
    __shared__ float A_s[BLOCK_DIM];
    __shared__ float B_s[BLOCK_DIM];
    __shared__ float C_s[BLOCK_DIM];
    if (i < N)
    {
        A_s[t] = A[i];
        B_s[t] = B[i];
    }
    else
    {
        A_s[t] = 0.0f;
        B_s[t] = 0.0f;
    }
    if (i + BLOCK_DIM < N)
    {
        A_s[t] += A[i + BLOCK_DIM];
        B_s[t] += B[i + BLOCK_DIM];
    }
    C_s[t] = A_s[t] * B_s[t];
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
            C_s[t] += C_s[t + stride];
    }

    if (t == 0)
        atomicAdd(result, C_s[0]);
}

// A, B, result are device pointers
void solve(const float *A, const float *B, float *result, int N)
{
    dot_product<<<(N + 1023) / 1024, BLOCK_DIM>>>(A, B, result, N);
}