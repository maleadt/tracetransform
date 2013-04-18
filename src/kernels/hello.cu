// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

//
// Configuration
//

// Standard library
#include <iostream>

// Local
#include "../logger.hpp"
#include "../cudahelper/memory.hpp"

// Static parameters
const int N = 16; 
const int blocksize = 16;


//
// Kernels
//

__global__ void hello_kernel(char *a, int *b) 
{
        a[threadIdx.x] += b[threadIdx.x];
}


//
// Wrappers
//

void hello()
{
        char a[N] = "Hello \0\0\0\0\0\0";
        int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        clog(info) << a << std::endl;

        CUDAHelper::GlobalMemory<char> ad(N);
        ad.upload(a);

        CUDAHelper::GlobalMemory<int> bd(N);
        bd.upload(b);

        dim3 dimBlock(blocksize, 1);
        dim3 dimGrid(1, 1);
        hello_kernel<<<dimGrid, dimBlock>>>(ad.data(), bd.data());

        ad.download(a);
        clog(info) << a << std::endl;
}
