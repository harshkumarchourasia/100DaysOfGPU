#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev); 

        printf("Device %d: %s\n", dev, prop.name);
        if (prop.managedMemory)
        {
            printf("  ✅ Supports Unified Memory\n");
        }
        else
        {
            printf("  ❌ Does NOT support Unified Memory\n");
        }
    }

    return 0;
}