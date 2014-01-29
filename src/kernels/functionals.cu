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



////////////////////////////////////////////////////////////////////////////////
// Auxiliary
//

__global__ void findWeightedMedian_kernel(const float *input,
                                          const float *prescan, int *output) {
    // Shared memory
    extern __shared__ float temp[];

    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Fetch
    temp[row] = prescan[row + col * rows];
    __syncthreads();

    if (row > 0) {
        float threshold = temp[rows - 1] / 2;
        if (temp[row - 1] < threshold && temp[row] >= threshold)
            output[col] = row;
    }
}

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

    // Scan
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
    free(precalc->prescan);
    free(precalc->medians);
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
    free(precalc->real);
    free(precalc->imag);
    free(precalc->prescan);
    free(precalc->medians);
    free(precalc);
}


//
// T6
//

void TFunctional6(const CUDAHelper::GlobalMemory<float> *input,
                  CUDAHelper::GlobalMemory<float> *output, int a) {
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
    precalc->transformed =
        new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(rows, cols));

    return precalc;
}

__global__ void extractMedianDomain_kernel(const float *input,
                                             const int *medians,
                                             float *output) {
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

    // Copy the positive domain, set to 0 otherwise
    if (row >= median)
        output[row + col * rows] = input[row + col * rows];
    else
        output[row + col * rows] = 0;
}

// NOTE: this is just the weighted median kernel + value writeback
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
    // NOTE: this doesn't actually make the domain smaller but just sets values
    //       falling outside of it to 0, since this simplifies quite a lot
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->rows(), 1);
        extractMedianDomain_kernel <<<blocks, threads>>>
            (*input, *precalc->medians, *precalc->transformed);
        CUDAHelper::checkState();
    }

    // Sort the transformed data
    {
        const int rows_padded = pow2ge(input->rows());
        dim3 threads(1, rows_padded);
        dim3 blocks(input->rows(), 1);
        sortBitonic_kernel <<<blocks, threads, rows_padded * sizeof(float)>>>
            (*precalc->transformed, input->rows());
        CUDAHelper::checkState();
    }

    // Launch prefix sum kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*precalc->transformed, *precalc->prescan, SQRT);
        CUDAHelper::checkState();
    }

    // Launch T7 kernel
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        TFunctional7_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*precalc->transformed, *precalc->prescan, *output, a);
        CUDAHelper::checkState();
    }
}

void TFunctional7_destroy(TFunctional7_precalc_t *precalc) {
    free(precalc->prescan);
    free(precalc->medians);
    free(precalc->transformed);
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

    // Fetch
    if (row == 0)
        temp[row] = 0;
    else
        temp[row] = fabs(input[row + col * rows] - input[row + col * rows - 1]);
    __syncthreads();

    // Scan
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

__global__ void PFunctional2_kernel(const float *input, const int *medians,
                                    float *output, int rows) {
    // Compute the thread dimensions
    const int col = blockIdx.x;

    // This is almost useless, isn't it
    output[col] = input[medians[col] + col * rows];
}

void PFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                  CUDAHelper::GlobalMemory<float> *output) {
    // Launch prefix sum kernel
    // TODO: precalculate
    CUDAHelper::GlobalMemory<float> *prescan =
        new CUDAHelper::GlobalMemory<float>(input->sizes());
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        prescan_kernel <<<blocks, threads, 2 * input->rows() * sizeof(float)>>>
            (*input, *prescan, NONE);
        CUDAHelper::checkState();
    }

    // Launch weighted median kernel
    // TODO: precalculate
    CUDAHelper::GlobalMemory<int> *medians = new CUDAHelper::GlobalMemory<int>(
        CUDAHelper::size_1d(input->cols()), 0);
    {
        dim3 threads(1, input->rows());
        dim3 blocks(input->cols(), 1);
        findWeightedMedian_kernel
                <<<blocks, threads, input->rows() * sizeof(float)>>>
            (*input, *prescan, *medians);
        CUDAHelper::checkState();
    }
    delete prescan;

    // Launch P2 kernel
    {
        dim3 threads(1, 1);
        dim3 blocks(input->cols(), 1);
        PFunctional2_kernel <<<blocks, threads>>>
            (*input, *medians, *output, input->rows());
        CUDAHelper::checkState();
    }
    delete medians;
}

//
// P3
//

void PFunctional3(const CUDAHelper::GlobalMemory<float> *,
                  CUDAHelper::GlobalMemory<float> *) {}

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
