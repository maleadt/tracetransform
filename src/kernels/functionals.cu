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

// TODO: needs decent synchronization, doesn't work cross-warp
__global__ void prescan_kernel(const float *_input,
                const int rows, const int cols,
                float *_output)
{
        // Shared memory
        extern __shared__ float temp[];

        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int row = threadIdx.y;

        // Construct Eigen objects
        Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
        Eigen::Map<Eigen::MatrixXf> output(_output, rows, cols);

        int thid = threadIdx.y;
        int n = rows;
        int pout = 0, pin = 1;

        // Do we need to do stuff?
        if (row < rows && col < cols) {
                // Load input into shared memory
                temp[pout * n + thid] = input(thid, col);
                __syncthreads();

                for (int offset = 1; offset < n; offset *= 2) {
                        // Swap double buffer indices
                        pout = 1 - pout;
                        pin = 1 - pin;
                        if (thid >= offset)
                                temp[pout * n + thid] = temp[pin * n + thid]
                                                + temp[pin * n + thid - offset];
                        else
                                temp[pout * n + thid] = temp[pin * n + thid];
                        __syncthreads();

                }

                // Write output
                output(thid, col) = temp[pout * n + thid];
        }
}

__global__ void findWeighedMedian_kernel(const float *_input, const float *_prescan,
                const int rows, const int cols,
                float *_output)
{
        // Compute the thread dimensions
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Construct Eigen objects
        Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
        Eigen::Map<const Eigen::MatrixXf> prescan(_prescan, rows, cols);
        Eigen::Map<Eigen::VectorXf> output(_output, cols);

        // Do we need to do stuff?
        if (col < cols) {
                for (int row = 0; row < rows; row++) {
                        if (2*prescan(row, col) >= prescan(rows-1, col)) {
                                output(col) = row;
                                break;
                        }
                }
        }
}

__global__ void TFunctionalRadon_kernel(const float *_input,
                const int rows, const int cols,
                float *_output, const int a)
{
        // Compute the thread dimensions
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < rows && col < cols) {
                // Construct Eigen objects
                Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
                Eigen::Map<Eigen::MatrixXf> output(_output, cols, 360);

                // Integrate
                atomicAdd(&output(col, a), input(row, col));
        }
}

__global__ void TFunctional1_kernel(const float *_input,
                const int rows, const int cols, const float *_medians,
                float *_output, const int a)
{
        // Compute the thread dimensions
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (col < cols) {
                // Construct Eigen objects
                Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
                Eigen::Map<const Eigen::VectorXf> medians(_medians, cols);
                Eigen::Map<Eigen::MatrixXf> output(_output, cols, 360);

                // Integrate
                const int median = medians(col);
                if (row < rows-median)
                        atomicAdd(&output(col, a), input(row+median, col)*row);
        }
}

//
// T functionals
//

void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, int a)
{
        assert(rows*cols == input->size());

        // Set-up
        CUDAHelper::Chrono chrono;
        chrono.start();

        // Launch radon kernel
        {
                dim3 threads(blocksize, blocksize);
                dim3 blocks(std::ceil((float)cols/blocksize), std::ceil((float)rows/blocksize));
                TFunctionalRadon_kernel<<<blocks, threads>>>(*input, rows, cols, *output, a);
                CUDAHelper::checkState();
        }

        // Clean-up
        chrono.stop();
        clog(trace) << "Radon kernel took " << chrono.elapsed() << " ms."
                        << std::endl;
}

void TFunctional1(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, int a)
{
        assert(rows*cols == input->size());

        // Set-up
        CUDAHelper::Chrono chrono;
        chrono.start();

        // Launch prefix sum kernel
        CUDAHelper::GlobalMemory<float> *prescan = new CUDAHelper::GlobalMemory<float>(rows*cols);
        {
                dim3 threads(1, rows);
                dim3 blocks(cols, 1);
                prescan_kernel<<<blocks, threads, 2*rows*sizeof(float)>>>(*input, rows, cols, *prescan);
                CUDAHelper::checkState();
        }

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<float> *medians = new CUDAHelper::GlobalMemory<float>(cols, 0);
        {
                dim3 threads(blocksize, 1);
                dim3 blocks(std::ceil((float)rows/blocksize), 1);
                findWeighedMedian_kernel<<<blocks, threads>>>(*input, *prescan, rows, cols, *medians);
                CUDAHelper::checkState();
        }
        delete prescan;

        // Launch T1 kernel
        {
                dim3 threads(blocksize, blocksize);
                dim3 blocks(std::ceil((float)cols/blocksize), std::ceil((float)rows/blocksize));
                TFunctional1_kernel<<<threads, blocks>>>(*input, rows, cols, *medians, *output, a);
                CUDAHelper::checkState();
        }
        delete medians;

        // Clean-up
        chrono.stop();
        clog(trace) << "T1 kernel took " << chrono.elapsed() << " ms."
                        << std::endl;
}

void TFunctional2(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, int a)
{

}

void TFunctional3(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, int a)
{

}

void TFunctional4(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, int a)
{

}

void TFunctional5(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, int a)
{

}


//
// P-functionals
//

CUDAHelper::GlobalMemory<float> *PFunctional1(
                const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols)
{

}

CUDAHelper::GlobalMemory<float> *PFunctional2(
                const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols)
{

}

CUDAHelper::GlobalMemory<float> *PFunctional3(
                const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols)
{

}

CUDAHelper::GlobalMemory<float> *PFunctionalHermite(
                const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, unsigned int order, int center)
{

}
