//
// Configuration
//

// Header
#include "nos.hpp"

// Standard library
#include <iostream>

// CULA
#include <cula_blas_device.hpp>
#include <cula_lapack_device.hpp>

// Local
#include "../logger.hpp"
#include "functionals.cu"


//
// Kernels
//

__global__ void offset_kernel(const float *input, const int *medians,
                              const int sinogram_center, int *output) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = threadIdx.y;
    const int cols = blockDim.y;

    // Calculate column offset
    int offset = medians[col] - sinogram_center;
    output[col] = offset;

    // Calculate minimum
    temp[col] = offset;
    // printf("Offset for col %d: %d\n", col, offset);
    __syncthreads();
    scan_array(temp, col, cols, MIN);
    if (col == cols - 1)
        output[cols] = temp[cols + col];

    // Calculate maximum
    temp[col] = offset;
    __syncthreads();
    scan_array(temp, col, cols, MAX);
    if (col == cols - 1)
        output[cols + 1] = temp[cols + col];
}

__global__ void alignment_kernel(const float *input, const int *offsets,
                                 const int padding, const int max,
                                 float *output) {
    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int cols = gridDim.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    output[max + row - offsets[col] + col * (rows + padding)] =
        input[row + col * rows];
}

__global__ void identity_kernel(float *output) {
    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int cols = gridDim.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    if (row == col)
        output[row + col * rows] = 1;
    else
        output[row + col * rows] = 0;
}


//
// Wrappers
//

CUDAHelper::GlobalMemory<float> *
nearest_orthonormal_sinogram(const CUDAHelper::GlobalMemory<float> *input,
                             size_t &new_center) {
    int rows = input->size(0);
    const int cols = input->size(1);

    // Launch prefix sum kernel
    CUDAHelper::GlobalMemory<float> *prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    {
        dim3 threads(1, rows);
        dim3 blocks(cols, 1);
        prescan_kernel << <blocks, threads, 2 * rows * sizeof(float)>>>
            (*input, *prescan, NONE);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    CUDAHelper::GlobalMemory<int> *medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);
    {
        dim3 threads(1, rows);
        dim3 blocks(cols, 1);
        findWeightedMedian_kernel << <blocks, threads, rows * sizeof(float)>>>
            (*input, *prescan, *medians);
        CUDAHelper::checkState();
    }
    delete prescan;

    // Launch offset kernel
    int sinogram_center = std::floor((rows - 1) / 2.0);
    CUDAHelper::GlobalMemory<int> *offsets =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols + 2));
    {
        dim3 threads(1, cols);
        dim3 blocks(1, 1);
        offset_kernel << <blocks, threads, 2 * cols * sizeof(float)>>>
            (*input, *medians, sinogram_center, *offsets);
    }
    int min, max;
    CUDAHelper::checkError(
        cudaMemcpy(&min, *offsets + cols, sizeof(int), cudaMemcpyDeviceToHost));
    CUDAHelper::checkError(cudaMemcpy(&max, *offsets + cols + 1, sizeof(int),
                                      cudaMemcpyDeviceToHost));
    delete medians;

    // Launch alignment kernel
    int padding = (int)(std::abs(max) + std::abs(min));
    new_center = sinogram_center + max;
    CUDAHelper::GlobalMemory<float> *aligned =
        new CUDAHelper::GlobalMemory<float>(
            CUDAHelper::size_2d(rows + padding, cols), 0);
    {
        dim3 threads(1, rows);
        dim3 blocks(cols, 1);
        alignment_kernel << <blocks, threads>>>
            (*input, *offsets, padding, max, *aligned);
        CUDAHelper::checkState();
    }
    rows += padding;
    delete offsets;

    // Calculate the SVD
    CUDAHelper::GlobalMemory<float> *vectorS =
        new CUDAHelper::GlobalMemory<float>(
            CUDAHelper::size_1d(std::min(rows, cols)));
    CUDAHelper::GlobalMemory<float> *matrixU =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, rows));
    CUDAHelper::GlobalMemory<float> *matrixVT =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(cols, cols));
    {
        culaDeviceSgesvd('A', 'A', rows, cols, *aligned, rows, *vectorS,
                         *matrixU, rows, *matrixVT, cols);
    }
    delete aligned;
    delete vectorS;

    // Calculate the NOS, cols)));
    // TODO: alias rows/cols for size(0) and size(1)
    CUDAHelper::GlobalMemory<float> *nos = new CUDAHelper::GlobalMemory<float>(
        CUDAHelper::size_2d(matrixU->size(0), matrixVT->size(1)));
    {
        assert(cols <= rows);
        culaDeviceSgemm('N', 'N', rows, cols, cols, 1, *matrixU, rows,
                        *matrixVT, cols, 0, *nos, rows);
    }
    delete matrixU;
    delete matrixVT;
    return nos;
}
