//
// Configuration
//

// Header
#include "functionals.hpp"

// Standard library
#include <stdio.h>

// Local
#include "../logger.hpp"
#include "scan.cu"
#include "sort.cu"

// TODO: why are all precalc->median arrays memsetted to 0?
// TODO: document generic functional kernels (not "launch P3 kernel")



////////////////////////////////////////////////////////////////////////////////
// Auxiliary
//

#ifdef WITH_CULA
__device__ float hermite_polynomial(unsigned int order, float x) {
    switch (order) {
    case 0:
        return 1.0;

    case 1:
        return 2.0 * x;

    default:
        return 2.0 * x * hermite_polynomial(order - 1, x) -
               2.0 * (order - 1) * hermite_polynomial(order - 2, x);
    }
}

__device__ unsigned int factorial(unsigned int n) {
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

__device__ float hermite_function(unsigned int order, float x) {
    return hermite_polynomial(order, x) /
           (sqrt(pow(2.0, (double)order) * factorial(order) * sqrt(M_PI)) *
            exp(x * x / 2));
}
#endif

unsigned int pow2ge(unsigned int n) {
    unsigned int k = 1;
    while (k < n)
        k *= 2;
    return k;
}



////////////////////////////////////////////////////////////////////////////////
// T-functionals
//

//
// Radon
//

__global__ void TFunctionalRadon_kernel(const float *input, float *output,
                                        const int a) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    temp[row] = input[row + col * rows];
    __syncthreads();

    // Scan to integrate
    scan_array(temp, row, rows, SUM);

    // Write back
    if (row == rows - 1)
        output[col + a * rows] = temp[rows + row];
}

void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input,
                      CUDAHelper::GlobalMemory<float> *output, int a) {
    // Launch radon kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctionalRadon_kernel
                <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *output, a);
        CUDAHelper::checkState();
    }
}


//
// T1
//

TFunctional12_precalc_t *TFunctional12_prepare(size_t rows, size_t cols) {
    TFunctional12_precalc_t *precalc =
        (TFunctional12_precalc_t *)malloc(sizeof(TFunctional12_precalc_t));

    precalc->prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);

    return precalc;
}

__global__ void TFunctional1_kernel(const float *input, const int *medians,
                                    float *output, const int a) {
    // Shared memory
    extern __shared__ float temp[];
    __shared__ int median;

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    if (row == 0)
        median = medians[col];
    __syncthreads();
    if (row < rows - median)
        temp[row] = input[row + median + col * rows] * row;
    else
        temp[row] = 0;
    __syncthreads();

    // Scan
    scan_array(temp, row, rows, SUM);

    // Write back
    if (row == rows - 1)
        output[col + a * rows] = temp[rows + row];
}

void TFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                  TFunctional12_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output, int a) {
    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, NONE);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        findWeightedMedian_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, *precalc->medians);
        CUDAHelper::checkState();
    }

    // Launch T1 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional1_kernel
                <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->medians, *output, a);
        CUDAHelper::checkState();
    }
}

void TFunctional12_destroy(TFunctional12_precalc_t *precalc) {
    delete precalc->prescan;
    delete precalc->medians;
    free(precalc);
}


//
// T2
//

__global__ void TFunctional2_kernel(const float *input, const int *medians,
                                    float *output, const int a) {
    // Shared memory
    extern __shared__ float temp[];
    __shared__ int median;

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    if (row == 0)
        median = medians[col];
    __syncthreads();
    if (row < rows - median)
        temp[row] = input[row + median + col * rows] * row * row;
    else
        temp[row] = 0;
    __syncthreads();

    // Scan
    scan_array(temp, row, rows, SUM);

    // Write back
    if (row == rows - 1)
        output[col + a * rows] = temp[rows + row];
}

void TFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                  TFunctional12_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output, int a) {
    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, NONE);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        findWeightedMedian_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, *precalc->medians);
        CUDAHelper::checkState();
    }

    // Launch T2 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional2_kernel
                <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->medians, *output, a);
        CUDAHelper::checkState();
    }
}


//
// T3, T4 and T5
//

__global__ void TFunctional345_kernel(const float *input, const int *medians,
                                      const float *precalc_real,
                                      const float *precalc_imag, float *output,
                                      const int a) {
    // Shared memory
    extern __shared__ float temp[];
    __shared__ int median;

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch index
    if (row == 0)
        median = medians[col];
    __syncthreads();

    // Fetch real part
    if (row > 0 && row < rows - median)
        temp[row] = precalc_real[row] * input[row + median + col * rows];
    else
        temp[row] = 0;
    __syncthreads();

    // Scan
    scan_array(temp, row, rows, SUM);

    // Write temporary
    float real;
    if (row == rows - 1)
        real = temp[rows + row];

    // Fetch imaginary part
    if (row > 0 && row < rows - median)
        temp[row] = precalc_imag[row] * input[row + median + col * rows];
    else
        temp[row] = 0;
    __syncthreads();

    // Scan
    scan_array(temp, row, rows, SUM);

    // Write back
    if (row == rows - 1) {
        float imag = temp[rows + row];
        output[col + a * rows] = hypot(real, imag);
    }
}

TFunctional345_precalc_t *TFunctional3_prepare(size_t rows, size_t cols) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    float *real_data = (float *)malloc(rows * sizeof(float));
    float *imag_data = (float *)malloc(rows * sizeof(float));

    for (unsigned int r = 1; r < rows; r++) {
        real_data[r] = r * cos(5.0 * log(r));
        imag_data[r] = r * sin(5.0 * log(r));
    }

    precalc->real =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(rows));
    precalc->imag =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(rows));

    precalc->real->upload(real_data);
    precalc->imag->upload(imag_data);

    free(real_data);
    free(imag_data);

    precalc->prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);

    return precalc;
}

TFunctional345_precalc_t *TFunctional4_prepare(size_t rows, size_t cols) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    float *real_data = (float *)malloc(rows * sizeof(float));
    float *imag_data = (float *)malloc(rows * sizeof(float));

    for (unsigned int r = 1; r < rows; r++) {
        real_data[r] = cos(3.0 * log(r));
        imag_data[r] = sin(3.0 * log(r));
    }

    precalc->real =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(rows));
    precalc->imag =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(rows));

    precalc->real->upload(real_data);
    precalc->imag->upload(imag_data);

    free(real_data);
    free(imag_data);

    precalc->prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);

    return precalc;
}

TFunctional345_precalc_t *TFunctional5_prepare(size_t rows, size_t cols) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    float *real_data = (float *)malloc(rows * sizeof(float));
    float *imag_data = (float *)malloc(rows * sizeof(float));

    for (unsigned int r = 1; r < rows; r++) {
        real_data[r] = sqrt(r) * cos(4.0 * log(r));
        imag_data[r] = sqrt(r) * sin(4.0 * log(r));
    }

    precalc->real =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(rows));
    precalc->imag =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(rows));

    precalc->real->upload(real_data);
    precalc->imag->upload(imag_data);

    free(real_data);
    free(imag_data);

    precalc->prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);

    return precalc;
}

void TFunctional345(const CUDAHelper::GlobalMemory<float> *input,
                    TFunctional345_precalc_t *precalc,
                    CUDAHelper::GlobalMemory<float> *output, int a) {
    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, SQRT);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        findWeightedMedian_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, *precalc->medians);
        CUDAHelper::checkState();
    }

    // Launch T345 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional345_kernel
                <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->medians, *precalc->real, *precalc->imag, *output,
             a);
        CUDAHelper::checkState();
    }
}

void TFunctional345_destroy(TFunctional345_precalc_t *precalc) {
    delete precalc->real;
    delete precalc->imag;
    delete precalc->prescan;
    delete precalc->medians;
    free(precalc);
}


//
// T6
//

TFunctional6_precalc_t *TFunctional6_prepare(size_t rows, size_t cols) {
    TFunctional6_precalc_t *precalc =
        (TFunctional6_precalc_t *)malloc(sizeof(TFunctional6_precalc_t));
    precalc->prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);
    precalc->extracted =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->weighted =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->indices =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_2d(rows, cols));
    precalc->permuted =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));

    return precalc;
}

// Extract the positive domain specified by the median, and weigh it
// NOTE: this just sets [0,median[ to 0, which we account for in other kernels
__global__ void TFunctional6_extract_kernel(const float *input,
                                            const int *medians, float *output1,
                                            float *output2) {
    // Shared memory
    __shared__ int median;

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch index
    if (row == 0)
        median = medians[col];
    __syncthreads();

    // Copy (and optionally weight) the positive domain, set to 0 otherwise
    if (row >= median) {
        float value = input[row + col * rows];
        output1[row + col * rows] = value;
        output2[row + col * rows] = value * (row - median);
    } else {
        output1[row + col * rows] = 0;
        output2[row + col * rows] = 0;
    }
}

// Permute the unweighted input based on sorting indices of the weighted input
__global__ void TFunctional6_permute_kernel(const float *input,
                                            const int *indices,
                                            const int *medians, float *output) {
    // Shared memory
    __shared__ int median;

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch index
    if (row == 0)
        median = medians[col];
    __syncthreads();

    // Permute the array
    if (row >= median) {
        int permuted_row = indices[row + col * rows];
        output[row + col * rows] = input[permuted_row + col * rows];
    } else {
        output[row + col * rows] = 0;
    }
}

// TODO: this is just findWeighedMedian + value writeback
__global__ void TFunctional6_kernel(const float *input, const float *prescan,
                                    float *output, const int a) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    temp[row] = prescan[row + col * rows];
    __syncthreads();

    // Find median value
    if (row > 0) {
        float threshold = temp[rows - 1] / 2;
        if (temp[row - 1] < threshold && temp[row] >= threshold) {
            output[col + a * rows] = input[row + col * rows];
        }
    }
}

void TFunctional6(const CUDAHelper::GlobalMemory<float> *input,
                  TFunctional6_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output, int a) {
    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, SQRT);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        findWeightedMedian_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, *precalc->medians);
        CUDAHelper::checkState();
    }

    // Extract and weight the positive domain of r
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional6_extract_kernel <<<blocks, threads>>>
            (*input, *precalc->medians, *precalc->extracted,
             *precalc->weighted);
        CUDAHelper::checkState();
    }

    // Sort the weighted data, and keep the indices
    {
        const int rows_padded = pow2ge(input->rows());
        dim3 threads(1, rows_padded);
        dim3 blocks(input->cols(), 1);
        sortBitonic_kernel
                <<<blocks, threads, 2 * rows_padded * sizeof(float)>>>
            (*precalc->weighted, input->rows(), *precalc->indices);
        CUDAHelper::checkState();
    }

    // Permute
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional6_permute_kernel <<<blocks, threads>>>
            (*precalc->extracted, *precalc->indices, *precalc->medians,
             *precalc->permuted);
        CUDAHelper::checkState();
    }

    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*precalc->permuted, *precalc->prescan, SQRT);
        CUDAHelper::checkState();
    }

    // Launch T6 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional6_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*precalc->weighted, *precalc->prescan, *output, a);
        CUDAHelper::checkState();
    }
}

void TFunctional6_destroy(TFunctional6_precalc_t *precalc) {
    delete precalc->prescan;
    delete precalc->medians;
    delete precalc->extracted;
    delete precalc->weighted;
    delete precalc->indices;
    delete precalc->permuted;
    free(precalc);
}


//
// T7
//

TFunctional7_precalc_t *TFunctional7_prepare(size_t rows, size_t cols) {
    TFunctional7_precalc_t *precalc =
        (TFunctional7_precalc_t *)malloc(sizeof(TFunctional7_precalc_t));
    precalc->prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);
    precalc->extracted =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));

    return precalc;
}

// Extract the positive domain specified by the median
// NOTE: this just sets [0,median[ to 0, which we account for in other kernels
__global__ void TFunctional7_extract_kernel(const float *input,
                                            const int *medians, float *output) {
    // Shared memory
    __shared__ int median;

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch index
    if (row == 0)
        median = medians[col];
    __syncthreads();

    // Copy (and optionally weight) the positive domain, set to 0 otherwise
    if (row >= median) {
        output[row + col * rows] = input[row + col * rows];
    } else
        output[row + col * rows] = 0;
}

// TODO: this is just findWeighedMedian + value writeback
__global__ void TFunctional7_kernel(const float *input, const float *prescan,
                                    float *output, const int a) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    temp[row] = prescan[row + col * rows];
    __syncthreads();

    // Find median value
    if (row > 0) {
        float threshold = temp[rows - 1] / 2;
        if (temp[row - 1] < threshold && temp[row] >= threshold)
            output[col + a * rows] = input[row + col * rows];
    }
}

void TFunctional7(const CUDAHelper::GlobalMemory<float> *input,
                  TFunctional7_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output, int a) {
    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, NONE);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        findWeightedMedian_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, *precalc->medians);
        CUDAHelper::checkState();
    }

    // Extract the positive domain of r
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional7_extract_kernel <<<blocks, threads>>>
            (*input, *precalc->medians, *precalc->extracted);
        CUDAHelper::checkState();
    }

    // Sort the extracted data
    {
        const int rows_padded = pow2ge(input->rows());
        dim3 threads(1, rows_padded);
        dim3 blocks(input->cols(), 1);
        sortBitonic_kernel <<<blocks, threads, rows_padded * sizeof(float)>>>
            (*precalc->extracted, input->rows(), NULL);
        CUDAHelper::checkState();
    }

    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*precalc->extracted, *precalc->prescan, SQRT);
        CUDAHelper::checkState();
    }

    // Launch T7 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional7_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*precalc->extracted, *precalc->prescan, *output, a);
        CUDAHelper::checkState();
    }
}

void TFunctional7_destroy(TFunctional7_precalc_t *precalc) {
    delete precalc->prescan;
    delete precalc->medians;
    delete precalc->extracted;
    free(precalc);
}


////////////////////////////////////////////////////////////////////////////////
// P-functionals
//

//
// P1
//

__global__ void PFunctional1_kernel(const float *input, float *output) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch and differentiate
    if (row == 0)
        temp[row] = 0;
    else
        temp[row] = fabs(input[row + col * rows] - input[row + col * rows - 1]);
    __syncthreads();

    // Scan to integrate
    scan_array(temp, row, rows, SUM);

    // Write back
    if (row == rows - 1)
        output[col] = temp[rows + row];
}

void PFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                  CUDAHelper::GlobalMemory<float> *output) {
    // Launch P1 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        PFunctional1_kernel
                <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *output);
        CUDAHelper::checkState();
    }
}

//
// P2
//

PFunctional2_precalc_t *PFunctional2_prepare(size_t rows, size_t cols) {
    PFunctional2_precalc_t *precalc =
        (PFunctional2_precalc_t *)malloc(sizeof(PFunctional2_precalc_t));
    precalc->prescan =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->medians =
        new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(cols), 0);

    return precalc;
}

__global__ void PFunctional2_kernel(const float *input, const int *medians,
                                    float *output, int rows) {
    // Compute the thread dimensions
    const int col = blockIdx.x;

    // This is almost useless, isn't it
    output[col] = input[medians[col] + col * rows];
}

void PFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                  PFunctional2_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output) {
    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, NONE);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        findWeightedMedian_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *precalc->prescan, *precalc->medians);
        CUDAHelper::checkState();
    }

    // Launch P2 kernel
    {
        dim3 threads(1, 1);
        dim3 blocks(input->cols(), 1);
        PFunctional2_kernel <<<blocks, threads>>>
            (*input, *precalc->medians, *output, input->rows());
        CUDAHelper::checkState();
    }
}

void PFunctional2_destroy(PFunctional2_precalc_t *precalc) {
    delete precalc->prescan;
    delete precalc->medians;
    free(precalc);
}

//
// P3
//

PFunctional3_precalc_t *PFunctional3_prepare(size_t rows, size_t cols) {
    PFunctional3_precalc_t *precalc =
        (PFunctional3_precalc_t *)malloc(sizeof(PFunctional3_precalc_t));
    precalc->real =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));
    precalc->imag =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));

    return precalc;
}

__global__ void PFunctional3_dft_kernel(const float *input, float *real,
                                        float *imag) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    temp[row] = input[row + col * rows];
    __syncthreads();

    // Calculate the DFT
    // TODO: precalculate arg/sinarg/cosarg?
    float local_real = 0, local_imag = 0;
    float arg = -2.0 * M_PI * (float)row / (float)rows;
    for (size_t i = 0; i < rows; i++) {
        float sinarg, cosarg;
        sincosf(i * arg, &sinarg, &cosarg);
        local_real += temp[i] * cosarg;
        local_imag += temp[i] * sinarg;
    }

    // Write back
    real[row + col * rows] = local_real;
    imag[row + col * rows] = local_imag;
}

__global__ void PFunctional3_kernel(const float *real, const float *imag,
                                    float *output) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Offsets into shared memory
    float *linspace = &temp[0];
    float *modifier = &temp[rows];
    float *trapz = &temp[2 * rows]; // 2*rows long for scan()

    // Fetch and transform
    linspace[row] = pow(
        hypot(real[row + col * rows] / rows, imag[row + col * rows] / rows), 4);
    modifier[row] = -1 + row * 2.0 / (rows - 1);
    __syncthreads();

    // Differentiate
    if (row == rows - 1)
        trapz[row] = 0;
    else
        trapz[row] = (linspace[row + 1] - linspace[row]) *
                     (modifier[row + 1] + modifier[row]);
    __syncthreads();

    // Scan to integrate
    scan_array(trapz, row, rows, SUM);

    // Write back
    if (row == rows - 1)
        output[col] = trapz[rows + row];
}

void PFunctional3(const CUDAHelper::GlobalMemory<float> *input,
                  PFunctional3_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output) {
    // Calculate the DFT
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        PFunctional3_dft_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *precalc->real, *precalc->imag);
        CUDAHelper::checkState();
    }

    // Launch the P3 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        PFunctional3_kernel
                <<<blocks, threads, 4 * input->rows() * sizeof(float)>>>
            (*precalc->real, *precalc->imag, *output);
        CUDAHelper::checkState();
    }
}

void PFunctional3_destroy(PFunctional3_precalc_t *precalc) {
    delete precalc->real;
    delete precalc->imag;
    free(precalc);
}

#ifdef WITH_CULA

//
// Hermite P-functionals
//

__global__ void PFunctionalHermite_kernel(const float *input, float *output,
                                          unsigned int order, int center) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Discretize the [-10, 10] domain to fit the column iterator
    float stepsize_lower = 10.0 / center;
    float stepsize_upper = 10.0 / (rows - 1 - center);

    // Calculate z
    // TODO: debug with test_svd
    float z;
    if ((row - 1) < center)
        z = row * stepsize_lower - 10;
    else
        z = (row - center) * stepsize_upper;

    // Fetch
    temp[row] = input[row + col * rows] * hermite_function(order, z);
    __syncthreads();

    // Scan
    scan_array(temp, row, rows, SUM);

    // Write back
    if (row == rows - 1)
        output[col] = temp[rows + row];
}

void PFunctionalHermite(const CUDAHelper::GlobalMemory<float> *input,
                        CUDAHelper::GlobalMemory<float> *output,
                        unsigned int order, int center) {
    // Launch Hermite kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        PFunctionalHermite_kernel
                <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *output, order, center);
        CUDAHelper::checkState();
    }
}

#endif
