#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define BLOCK_DIM_2D 32
#define TILE_WIDTH 32
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

         for (int k = 0; k < TILE_WIDTH; k++)
        {
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        __syncthreads();
    }

     if (row < a1 && col < b2)
        C[row * b2 + col] = value;
}

int main()
{
    int a1 = 2, a2 = 3, b2 = 1;
    float A[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    float B[3][1] = {{-1.0f}, {1.0f}, {1.0f}};

    float *d_A, *d_B, *d_C, *C;

    C = (float*)malloc(2 * sizeof(float));

    cudaMalloc((void **)&d_A, 6 * sizeof(float));
    cudaMalloc((void **)&d_B, 3 * sizeof(float));
    cudaMalloc((void **)&d_C, 2 * sizeof(float));

    cudaMemcpy(d_A, &A[0][0], 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &B[0][0], 6 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 grid_size((1 + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D, (2 + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D);
    dim3 block_size(BLOCK_DIM_2D, BLOCK_DIM_2D);
    mat_mul<<<grid_size, block_size>>>(d_A, d_B, d_C, a2, a1, b2);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    cudaMemcpy(C, d_C, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 1; j++)
        {
            cout << C[i + j] << " ";
        }
        cout << endl;
    }

    return 0;
}