//
// Configuration
//

// Header
#include "functionals.hpp"

// Local
#include "../global.hpp"
#include "../logger.hpp"
#include "../cudahelper/chrono.hpp"

// Static parameters
const int blocksize = 8;


//
// Kernels
//

__global__ void findWeighedMedian_kernel(const float *_input,
                const int rows, const int cols,
                float *_output)
{
        // Compute the thread dimensions
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Construct Eigen objects
        Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
        Eigen::Map<Eigen::VectorXf> output(_output, cols);

        // Do we need to do stuff?
        if (col < cols) {
                float sum = 0;
                for (int row = 0; row < rows; row++)
                        sum += input(row, col);

                float integral = 0;
                for (int row = 0; row < rows; row++) {
                        integral += input(row, col);
                        if (2*integral >= sum) {
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

        // Construct Eigen objects
        Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
        Eigen::Map<Eigen::MatrixXf> output(_output, cols, 360);

        // Do we need to do stuff?
        if (row < rows && col < cols)
                atomicAdd(&output(col, a), input(row, col));
}

__global__ void TFunctional1_kernel(const float *_input,
                const int rows, const int cols, const float *_medians,
                float *_output, const int a)
{
        // Compute the thread dimensions
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        // Construct Eigen objects
        Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
        Eigen::Map<const Eigen::VectorXf> medians(_medians, cols);
        Eigen::Map<Eigen::MatrixXf> output(_output, cols, 360);

        // Do we need to do stuff?
        if (col < cols) {
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
                dim3 blocks(std::ceil((float)rows/blocksize), std::ceil((float)cols/blocksize));
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

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<float> *medians = new CUDAHelper::GlobalMemory<float>(cols, 0);
        {
                dim3 threads(blocksize, 1);
                dim3 blocks(std::ceil((float)rows/blocksize), 1);
                findWeighedMedian_kernel<<<blocks, threads>>>(*input, rows, cols, *medians);
                CUDAHelper::checkState();
        }

        // Launch T1 kernel
        {
                dim3 threads(blocksize, blocksize);
                dim3 blocks(std::ceil((float)rows/blocksize), std::ceil((float)cols/blocksize));
                TFunctional1_kernel<<<threads, blocks>>>(*input, rows, cols, *medians, *output, a);
                CUDAHelper::checkState();
        }

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
