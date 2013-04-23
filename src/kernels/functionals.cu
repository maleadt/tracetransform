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
const int blocksize = 16;


//
// Kernels
//

__global__ void TFunctionalRadon_kernel(const float* _input,
                const unsigned int width, const unsigned int height,
                float* _output, const int a)
{
        // Compute the thread dimensions
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

        // Get Eigen matrices back
        Eigen::Map<const Eigen::MatrixXf> input(_input, height, width);
        Eigen::Map<Eigen::MatrixXf> output(_output, height, 360);

        // Do we need to do stuff?
        if (col < width && row < height) {
                atomicAdd(&output(row, a), input(row, col));
        }
}

//
// T functionals
//

void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, size_t a)
{
        assert(rows*cols == input->size());

        CUDAHelper::Chrono chrono;
        chrono.start();

        dim3 threads(blocksize, blocksize);
        dim3 blocks(rows / blocksize, cols / blocksize);
        TFunctionalRadon_kernel<<<blocks, threads>>>(*input, rows, cols, *output, a);

        chrono.stop();
        clog(trace) << "Radon kernel took " << chrono.elapsed() << " ms."
                        << std::endl;
}

void TFunctional1(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, size_t a)
{

}

void TFunctional2(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, size_t a)
{

}

void TFunctional3(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, size_t a)
{

}

void TFunctional4(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, size_t a)
{

}

void TFunctional5(const CUDAHelper::GlobalMemory<float> *input, int rows,
                int cols, CUDAHelper::GlobalMemory<float> *output, size_t a)
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
                int cols, unsigned int order, size_t center)
{

}
