#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define BLOCK_DIM_2D 32
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

////////////

__global__ void transpose(float *matrix, int n_row, int n_col, float *output)
{
    __shared__ float data_s[BLOCK_DIM_2D][BLOCK_DIM_2D];
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int i = bx + threadIdx.x;
    int j = by + threadIdx.y;
    if (i < n_col && j < n_row)
        data_s[threadIdx.y][threadIdx.x] = matrix[i + n_col * j];
    else
        data_s[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();
    int transposed_x = blockIdx.y * blockDim.y + threadIdx.x;
    int transposed_y = blockIdx.x * blockDim.x + threadIdx.y;

    if (transposed_x < n_row && transposed_y < n_col)
        output[transposed_y * n_row + transposed_x] = data_s[threadIdx.x][threadIdx.y];
}

int main()
{
    int row = 2;
    int col = 3;
    float matrix[row][col] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    int size = row * col * sizeof(float);

    float *result;
    result = (float *)malloc(size);

    cout << "Input Matrix" << endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    float *d_matrix, *d_result;
    gpuErrchk(cudaMalloc((void **)&d_matrix, size));
    gpuErrchk(cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc((void **)&d_result, size));

    dim3 block_dim(BLOCK_DIM_2D, BLOCK_DIM_2D);
    dim3 grid_size((col + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D, (row + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D);

    transpose<<<grid_size, block_dim>>>(d_matrix, row, col, d_result);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost));

    cout << "Output Matrix:" << endl;
    for (int i = 0; i < col; i++)
    {
        for (int j = 0; j < row; j++)
        {
            cout << result[i * row + j] << " ";
        }
        cout << endl;
    }

    return 0;
}