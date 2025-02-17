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
#include <stdio.h>
#include <time.h>

using namespace std;
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define BLOCK_DIM_1D 1024
#define BLOCK_DIM_2D 32
#define TILE_WIDTH 32


__global__ void get_sum(float *data, float *result, int nx)
{
    int data_id = blockIdx.y;
    int t = threadIdx.x;
    int i = data_id * nx + 2 * blockIdx.x * BLOCK_DIM_1D + t;
    __shared__ float data_s[BLOCK_DIM_1D];
    data_s[t] = 0.0f;
    __syncthreads();
    if (i < (1 + data_id) * nx)
        data_s[t] = data[i];
    if ((i + BLOCK_DIM_1D) < (1 + data_id) * nx)
        data_s[t] += data[i + BLOCK_DIM_1D];
    for (int stride = BLOCK_DIM_1D / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride && (t + stride) < BLOCK_DIM_1D)
        {
            data_s[t] = data_s[t] + data_s[t + stride];
        }
    }
    if (t == 0)
    {
        atomicAdd(&result[data_id], data_s[0]);
    }
}

__global__ void mean_center_and_transpose(float *matrix, int n_row, int n_col, float *output, float *sums)
{
    __shared__ float data_s[BLOCK_DIM_2D][BLOCK_DIM_2D];
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int col = bx + threadIdx.x;
    int row = by + threadIdx.y;
    if (col < n_col && row < n_row)
    {
        float val = matrix[col + n_col * row];
        val -= (__ldg(&sums[row])/n_col);
        data_s[threadIdx.y][threadIdx.x] = val;
        matrix[col + n_col * row] = val;
    }
    else
    {
        data_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    int transposed_col = blockIdx.y * blockDim.y + threadIdx.x;
    int transposed_row = blockIdx.x * blockDim.x + threadIdx.y;

    if (transposed_row < n_col && transposed_col < n_row)
        output[transposed_row * n_row + transposed_col] = data_s[threadIdx.x][threadIdx.y];
}

__global__ void mat_mul(float *A, float *B, float *C, int a2, int a1, int b2)
{
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = bx * blockDim.x + tx;  
    int row = by * blockDim.y + ty;  
    float value = 0;

     for (int phase = 0; phase < (a2 + TILE_WIDTH - 1) / TILE_WIDTH; phase++)
    {
         if (row < a1 && phase * TILE_WIDTH + tx < a2)
            sh_A[ty][tx] = A[row * a2 + phase * TILE_WIDTH + tx];
        else
            sh_A[ty][tx] = 0.0f;

         if (col < b2 && phase * TILE_WIDTH + ty < a2)
            sh_B[ty][tx] = B[(phase * TILE_WIDTH + ty) * b2 + col];
        else
            sh_B[ty][tx] = 0.0f;

        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        __syncthreads();
    }

     if (row < a1 && col < b2)
        C[row * b2 + col] = value;
}

__global__ void normalize_result(float *input, float *output, int ny){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= ny || col >= ny) return;
    output[row * ny + col] = input[row * ny + col] / sqrt(__ldg(&input[row*ny+ row]) * __ldg(&input[col*ny+col]) );
}

void correlate(int ny, int nx, const float *data, float *result)
{
    float *d_data, *d_sums, *d_result, *d_transpose; 
    gpuErrchk(cudaMalloc((void **)&d_data, nx * ny * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_transpose, nx * ny * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void **)&d_sums, ny * sizeof(float)));
    gpuErrchk(cudaMemset(d_sums, 0, ny * sizeof(float)));

    gpuErrchk(cudaMalloc((void **)&d_result, ny * ny * sizeof(float)));

    dim3 dimGrid((nx + 2 * BLOCK_DIM_1D - 1) / (2 * BLOCK_DIM_1D), ny);

    get_sum<<<dimGrid, BLOCK_DIM_1D>>>(d_data, d_sums, nx);

    dim3 grid_dim((nx + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D, (ny + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D);
    dim3 block_dim(BLOCK_DIM_2D, BLOCK_DIM_2D);
    mean_center_and_transpose<<<grid_dim, block_dim>>>(d_data, ny, nx, d_transpose, d_sums);
    
    dim3 dimGrid2((ny + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D, (ny + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D);
    mat_mul<<< dimGrid2,block_dim >>>(d_data, d_transpose, d_result, nx, ny, ny);

    // Allocate pinned memory on host
    float *d_norm_result;
    gpuErrchk(cudaMalloc((void**)&d_norm_result, ny * ny * sizeof(float)));

    normalize_result<<< dimGrid2,block_dim >>>(d_result, d_norm_result, ny);

    gpuErrchk(cudaMemcpy(result, d_norm_result, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    /*
    for(int i = 0; i < ny; i++){
        for(int j = 0; j < ny; j++){
            result[i*ny + j] = _result[i*ny + j] / sqrt(_result[j*ny + j] * _result[i*ny + i]);
        }
    }
    */

    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_result));
    gpuErrchk(cudaFree(d_sums));
    gpuErrchk(cudaFree(d_transpose));
    gpuErrchk(cudaFree(d_norm_result));

    // Free pinned memory
    // gpuErrchk(cudaFreeHost(_result));
}