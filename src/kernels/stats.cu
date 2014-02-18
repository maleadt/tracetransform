//
// Configuration
//

// Header
#include "stats.hpp"

// Local
#include "scan.cu"


//
// Kernels
//

__global__ void zscore_kernel(const float *input, float *output) {
    // Shared memory
    extern __shared__ double temp_hp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Offsets into shared memory
    double *mean = &temp_hp[2 * rows];
    double *stdev = &temp_hp[2 * rows + 1];

    // Fetch
    temp_hp[row] = input[row + col * rows];
    __syncthreads();

    // Scan to integrate
    scan_array(temp_hp, row, rows, SUM);

    // Calculate the arithmetic mean
    if (row == rows - 1)
        *mean = temp_hp[rows + row] / rows;
    __syncthreads();

    // Fetch and differentiate against mean
    double diff = input[row + col * rows] - *mean;
    temp_hp[row] = diff * diff;

    // Scan to integrate
    scan_array(temp_hp, row, rows, SUM);

    // Calculate the standard deviation
    if (row == rows - 1)
        *stdev = std::sqrt(temp_hp[rows + row] / (rows - 1));
    __syncthreads();

    // Normalize the input
    output[row + col * rows] = (input[row + col * rows] - *mean) / *stdev;
}


//
// Wrappers
//

CUDAHelper::GlobalMemory<float> *
zscore(const CUDAHelper::GlobalMemory<float> *input) {
    // Calculate the z-score
    CUDAHelper::GlobalMemory<float> *output =
        new CUDAHelper::GlobalMemory<float>(input->sizes());
    {
        dim3 threads(1, input->rows());
        dim3 blocks(1, 1);
        zscore_kernel <<<blocks, threads, 2 * input->rows() * sizeof(double) +
                                              2 * sizeof(double)>>>
            (*input, *output);
        CUDAHelper::checkState();
    }

    return output;
}
