// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

//
// Configuration
//

// Header
#include "hello.hpp"

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

__constant__ int b[N];
__global__ void hello_kernel(char *a)
{
        a[threadIdx.x] += b[threadIdx.x];
}


//
// Wrappers
//

void hello()
{
        char a_data[N] = "Hello \0\0\0\0\0\0";
        int b_data[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        clog(info) << a_data << std::endl;

        CUDAHelper::GlobalMemory<char> a_mem(N);
        a_mem.upload(a_data);

        CUDAHelper::ConstantMemory<int> b_mem(b, N);
        b_mem.upload(b_data);

        dim3 dimBlock(blocksize, 1);
        dim3 dimGrid(1, 1);
        hello_kernel<<<dimGrid, dimBlock>>>(a_mem);

        a_mem.download(a_data);
        clog(info) << a_data << std::endl;
}
