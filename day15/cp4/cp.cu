
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

#define BLOCK_DIM 1024

__global__ void get_sum(float *data, float *result, int nx)
{
    int data_id = blockIdx.y;
    int t = threadIdx.x;
    int i = data_id * nx + 2 * blockIdx.x * BLOCK_DIM + t;
    __shared__ float data_s[BLOCK_DIM];
    data_s[t] = 0.0f;
    __syncthreads();
    if (i < (1 + data_id) * nx)
        data_s[t] = data[i];
    if ((i + BLOCK_DIM) < (1 + data_id) * nx)
        data_s[t] = data[i] + data[i + BLOCK_DIM];
    for (int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (t < stride && (t + stride) < BLOCK_DIM)
        {
            data_s[t] = data_s[t] + data_s[t + stride];
        }
    }
    if (t == 0)
    {
        atomicAdd(&result[data_id], data_s[0]);
    }
}

int main()
{
    int nx = 2000;
    int ny = 5;
    float *data;
    data = (float *)malloc(nx * ny * sizeof(float));
    for (int row = 0; row < ny; row++)
    {
        for (int col = 0; col < nx; col++)
        {
            data[row * nx + col] = row * 1.0f;
        }
    }

    float *d_data, *d_sums, *sums;
    cudaMalloc((void **)&d_data, nx * ny * sizeof(float));
    cudaMemcpy(d_data, data, nx * ny * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_sums, ny * sizeof(float));

    dim3 dimGrid((nx + 2 * BLOCK_DIM - 1) / (2 * BLOCK_DIM), ny);

    get_sum<<<dimGrid, BLOCK_DIM>>>(d_data, d_sums, nx);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    sums = (float *)malloc(ny * sizeof(float));
    cudaMemcpy(sums, d_sums, ny * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_sums);
    delete[] data;
    for (int i = 0; i < ny; i++)
    {
        cout << "i: " << i << " sum: " << sums[i] << endl;
    }
}
