#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
        float x = input[i];
        float y = input[N-1-i];
        input[i] = y;
        input[N-1-i] = x;
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}