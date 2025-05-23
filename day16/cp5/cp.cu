
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

__global__ void corr(int ny, int nx, const float *data, float *result, float *sums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j > i || i >= ny || j >= ny)
        return;
    if (i == j)
    {
        result[i + j * ny] = 1.0f;
        return;
    }
    __shared__ float data_s_A[BLOCK_DIM_2D][BLOCK_DIM_2D];
    __shared__ float data_s_B[BLOCK_DIM_2D][BLOCK_DIM_2D];
    float avg_i = sums[i] / nx;
    float avg_j = sums[j] / nx;
    float sum_ij = 0.0f;
    float sum_i = 0.0f;
    float sum_j = 0.0f;
    for(int phase = 0; phase <= (nx+blockDim.x-1)/blockDim.x; phase++){
        data_s_A[threadIdx.x][threadIdx.y] = data[nx*i + phase * BLOCK_DIM_2D + threadIdx.x];
        data_s_B[threadIdx.x][threadIdx.y] = data[nx*j + phase * BLOCK_DIM_2D + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < BLOCK_DIM_2D; k++){
            sum_ij += (data_s_A[threadIdx.x][k] - avg_i) * (data_s_B[threadIdx.x][k] - avg_j);
            sum_i += (data_s_A[threadIdx.x][k] - avg_i) * (data_s_A[threadIdx.x][k] - avg_i);
            sum_j += (data_s_B[threadIdx.x][k] - avg_j) * (data_s_B[threadIdx.x][k] - avg_j);
        }
        __syncthreads();
    }
    result[i + j * ny] = sum_ij / sqrt(sum_i * sum_j);
    result[j + i * ny] = sum_ij / sqrt(sum_i * sum_j);
}

void correlate(int ny, int nx, const float *data, float *result)
{
    float *d_data, *d_sums, *d_result;
    gpuErrchk(cudaMalloc((void **)&d_data, nx * ny * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void **)&d_sums, ny * sizeof(float)));
    gpuErrchk(cudaMemset(d_sums, 0, ny * sizeof(float)));

    gpuErrchk(cudaMalloc((void **)&d_result, ny * ny * sizeof(float)));

    dim3 dimGrid((nx + 2 * BLOCK_DIM_1D - 1) / (2 * BLOCK_DIM_1D), ny);

    get_sum<<<dimGrid, BLOCK_DIM_1D>>>(d_data, d_sums, nx);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
    
    
    dim3 grid_dim((ny+31)/32, (ny+31)/32);
    dim3 block_dim(BLOCK_DIM_2D,BLOCK_DIM_2D);
    corr<<<grid_dim, block_dim>>>(ny, nx, d_data, d_result, d_sums);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(result, d_result, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_result));
    gpuErrchk(cudaFree(d_sums));
}