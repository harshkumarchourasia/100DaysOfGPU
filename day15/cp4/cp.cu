/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024

__global__ void get_sum(float *data, float *result, int nx){
    int data_id = blockIdx.y;
    int t = blockIdx.x;
    int i = 2 * blockIdx.x * BLOCK_DIM + t;
    __shared__ float data_s[BLOCK_DIM];
    data_s[t] = 0.0f;
    __syncthreads();
    if (i < nx and (i+BLOCK_DIM)<nx) data_s[t] = data[i] + data[i + BLOCK_DIM];
    for(int stride = BLOCK_DIM / 2; stride >= 1; stride/=2){
         __syncthreads();
         if (t < stride && (t+stride) < BLOCK_DIM)
        {
            data_s[t] = data_s[t] + data_s[t + stride];
        }
    }
    if(t==0){
        atomicAdd(&result[data_id], data_s[0]);
    }
}

__global__ void corr(int ny, int nx, float *data, float *result, float *sums){

}

void correlate(int ny, int nx, const float *data, float *result)
{
    float *d_data, *d_result, *d_sums;
    cudaMalloc((void **)&d_data, nx * ny * sizeof(float));
    cudaMalloc((void **)&d_result, ny * ny * sizeof(float));
    cudaMalloc((void **)&d_sums, ny * sizeof(float));
    cudaMemcpy(d_data, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((nx + 2 * BLOCK_DIM - 1) / (2 * BLOCK_DIM), ny);
    get_sum<<<dimGrid, dimBlock>>>(d_data, d_sums, nx);
    //corr<<<1, 1>>>(ny, nx, d_data, d_result, d_sums);

    cudaMemcpy(result, d_result, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_sums);
}
