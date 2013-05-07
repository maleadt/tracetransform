//
// Configuration
//

// Header
#include "functionals.hpp"

// Standard library
#include <stdio.h>

// Local
#include "../global.hpp"
#include "../logger.hpp"
#include "../cudahelper/chrono.hpp"

// Static parameters
const int blocksize = 8;


//
// Kernels
//

enum prescan_function_t {
        NONE = 0,
        SQRT
};

__global__ void prescan_kernel(const float *input,
                float *output, prescan_function_t prescan_function)
{
        // Shared memory
        extern __shared__ float temp[];

        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int cols = gridDim.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Load input into shared memory
        switch (prescan_function) {
                case SQRT:
                        temp[row] = sqrt(input[row + col*rows]);
                        break;
                default:
                        temp[row] = input[row + col*rows];
                        break;
        }
        __syncthreads();

        int pout = 0, pin = 1;
        for (int offset = 1; offset < rows; offset *= 2) {
                // Swap double buffer indices
                pout = 1 - pout;
                pin = 1 - pin;
                if (row >= offset)
                        temp[pout * rows + row] = temp[pin * rows + row]
                                        + temp[pin * rows + row - offset];
                else
                        temp[pout * rows + row] = temp[pin * rows + row];
                __syncthreads();

        }

        // Write output
        output[row + col*rows] = temp[pout * rows + row];
}

__global__ void findWeighedMedian_kernel(const float *input, const float *prescan,
                int *output)
{
        // Shared memory
        extern __shared__ float temp[];

        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int cols = gridDim.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Load input into shared memory
        temp[row] = prescan[row + col*rows];
        __syncthreads();

        if (row > 0) {
                float threshold = temp[rows-1] / 2;
                if (temp[row-1] < threshold && temp[row] >= threshold)
                        output[col] = row;
        }
}

__global__ void TFunctionalRadon_kernel(const float *input,
                float *output, const int a)
{
        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int cols = gridDim.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Integrate
        atomicAdd(&output[col + a*rows], input[row + col*rows]);
}

__global__ void TFunctional1_kernel(const float *input,
                const int *medians,
                float *output, const int a)
{
        // Shared memory
        extern __shared__ float temp[];
        __shared__ int median;

        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int cols = gridDim.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Fetch
        if (row == 0)
                median = medians[col];
        __syncthreads();
        if (row < rows-median)
                temp[row] = input[row+median + col*rows] * row;
        else
                temp[row] = 0;
        __syncthreads();

        // Reduce
        int pout = 0, pin = 1;
        for (int offset = 1; offset < rows; offset *= 2) {
                // Swap double buffer indices
                pout = 1 - pout;
                pin = 1 - pin;
                if (row >= offset)
                        temp[pout * rows + row] = temp[pin * rows + row]
                                        + temp[pin * rows + row - offset];
                else
                        temp[pout * rows + row] = temp[pin * rows + row];
                __syncthreads();

        }

        // Write back
        if (row == rows-1)
                output[col + a*rows] = temp[pout * rows + row];
}

__global__ void TFunctional2_kernel(const float *input,
                const int *medians,
                float *output, const int a)
{
        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int cols = gridDim.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Integrate
        const int median = medians[col];
        if (row < rows-median)
                atomicAdd(&output[col + a*rows], input[row+median + col*rows] * row * row);
}

//
// T functionals
//

void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{
        const int rows = input->size(0);
        const int cols = input->size(1);

        // Set-up
        CUDAHelper::Chrono chrono;
        chrono.start();

        // Launch radon kernel
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                TFunctionalRadon_kernel<<<blocks, threads>>>(*input, *output, a);
                CUDAHelper::checkState();
        }

        // Clean-up
        chrono.stop();
        clog(trace) << "Radon kernel took " << chrono.elapsed() << " ms."
                        << std::endl;
}

void TFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{
        const int rows = input->size(0);
        const int cols = input->size(1);

        // Set-up
        CUDAHelper::Chrono chrono;
        chrono.start();

        // Launch prefix sum kernel
        CUDAHelper::GlobalMemory<float> *prescan = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                prescan_kernel<<<blocks, threads, 2*rows*sizeof(float)>>>(*input, *prescan, NONE);
                CUDAHelper::checkState();
        }

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<int> *medians = new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                findWeighedMedian_kernel<<<blocks, threads, rows*sizeof(float)>>>(*input, *prescan, *medians);
                CUDAHelper::checkState();
        }
        delete prescan;

        // Launch T1 kernel
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                TFunctional1_kernel<<<blocks, threads, 2*rows*sizeof(float)>>>(*input, *medians, *output, a);
                CUDAHelper::checkState();
        }
        delete medians;

        // Clean-up
        chrono.stop();
        clog(trace) << "T1 kernel took " << chrono.elapsed() << " ms."
                        << std::endl;
}

void TFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{
        const int rows = input->size(0);
        const int cols = input->size(1);

        // Set-up
        CUDAHelper::Chrono chrono;
        chrono.start();

        // Launch prefix sum kernel
        CUDAHelper::GlobalMemory<float> *prescan = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                prescan_kernel<<<blocks, threads, 2*rows*sizeof(float)>>>(*input, *prescan, NONE);
                CUDAHelper::checkState();
        }

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<int> *medians = new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                findWeighedMedian_kernel<<<blocks, threads, rows*sizeof(float)>>>(*input, *prescan, *medians);
                CUDAHelper::checkState();
        }
        delete prescan;

        // Launch T2 kernel
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                TFunctional2_kernel<<<blocks, threads>>>(*input, *medians, *output, a);
                CUDAHelper::checkState();
        }
        delete medians;

        // Clean-up
        chrono.stop();
        clog(trace) << "T2 kernel took " << chrono.elapsed() << " ms."
                        << std::endl;
}

void TFunctional3(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{

}

void TFunctional4(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{

}

void TFunctional5(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{

}


//
// P-functionals
//

CUDAHelper::GlobalMemory<float> *PFunctional1(
                const CUDAHelper::GlobalMemory<float> *input)
{

}

CUDAHelper::GlobalMemory<float> *PFunctional2(
                const CUDAHelper::GlobalMemory<float> *input)
{

}

CUDAHelper::GlobalMemory<float> *PFunctional3(
                const CUDAHelper::GlobalMemory<float> *input)
{

}

CUDAHelper::GlobalMemory<float> *PFunctionalHermite(
                const CUDAHelper::GlobalMemory<float> *input,
                unsigned int order, int center)
{

}
