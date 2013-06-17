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


//
// Kernels
//

enum scan_operation_t {
        SUM = 0,
        MIN,
        MAX
};

// TODO: replace with faster tree-based algorithm
//       http://stackoverflow.com/questions/11385475/scan-array-cuda
__device__ void scan_array(float* temp, int index, int length, scan_operation_t operation)
{
        int pout = 0, pin = 1;
        for (int offset = 1; offset < length; offset *= 2) {
                // Swap double buffer indices
                pout = 1 - pout;
                pin = 1 - pin;
                if (index >= offset) {
                        switch (operation) {
                                case SUM:
                                        temp[pout * length + index] = temp[pin * length + index]
                                                        + temp[pin * length + index - offset];
                                        break;
                                case MIN:
                                        temp[pout * length + index] = fmin(temp[pin * length + index]
                                                        , temp[pin * length + index - offset]);
                                        break;
                                case MAX:
                                        temp[pout * length + index] = fmax(temp[pin * length + index]
                                                        , temp[pin * length + index - offset]);
                                        break;

                        }
                } else {
                        temp[pout * length + index] = temp[pin * length + index];
                }
                __syncthreads();

        }
        temp[pin * length + index] = temp[pout * length + index];
}

enum prescan_function_t {
        NONE = 0,
        SQRT
};

__global__ void prescan_kernel(const float *input, float *output,
                const prescan_function_t prescan_function)
{
        // Shared memory
        extern __shared__ float temp[];

        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Fetch
        switch (prescan_function) {
                case SQRT:
                        temp[row] = sqrt(input[row + col*rows]);
                        break;
                case NONE:
                default:
                        temp[row] = input[row + col*rows];
                        break;
        }
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write back
        output[row + col*rows] = temp[rows + row];
}

__global__ void findWeighedMedian_kernel(const float *input,
                const float *prescan, int *output)
{
        // Shared memory
        extern __shared__ float temp[];

        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Fetch
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
        // Shared memory
        extern __shared__ float temp[];

        // Compute the thread dimensions
        const int col = blockIdx.x;
        const int row = threadIdx.y;
        const int rows = blockDim.y;

        // Fetch
        temp[row] = input[row + col*rows];
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write back
        if (row == rows-1)
                output[col + a*rows] = temp[rows + row];
}

__global__ void TFunctional1_kernel(const float *input, const int *medians,
                float *output, const int a)
{
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
        if (row < rows-median)
                temp[row] = input[row+median + col*rows] * row;
        else
                temp[row] = 0;
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write back
        if (row == rows-1)
                output[col + a*rows] = temp[rows + row];
}

__global__ void TFunctional2_kernel(const float *input, const int *medians,
                float *output, const int a)
{
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
        if (row < rows-median)
                temp[row] = input[row+median + col*rows] * row * row;
        else
                temp[row] = 0;
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write back
        if (row == rows-1)
                output[col + a*rows] = temp[rows + row];
}

__global__ void TFunctional345_kernel(const float *input, const int *medians,
                const float *precalc_real, const float *precalc_imag,
                float *output, const int a)
{
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
        if (row > 0 && row < rows-median)
                temp[row] = precalc_real[row] * input[row+median + col*rows];
        else
                temp[row] = 0;
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write temporary
        float real;
        if (row == rows-1)
                real = temp[rows + row];

        // Fetch imaginary part
        if (row > 0 && row < rows-median)
                temp[row] = precalc_imag[row] * input[row+median + col*rows];
        else
                temp[row] = 0;
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write back
        if (row == rows-1) {
                float imag = temp[rows + row];
                output[col + a*rows] = hypot(real, imag);
        }
}

__global__ void PFunctional1_kernel(const float *input,
                float *output)
{
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
                temp[row] = fabs(input[row + col*rows] - input[row + col*rows - 1]);
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write back
        if (row == rows-1)
                output[col] = temp[rows + row];
}

__global__ void PFunctional2_kernel(const float *input, const int *medians,
                float *output, int rows)
{
        // Compute the thread dimensions
        const int col = blockIdx.x;

        // This is almost useless, isn't it
        output[col] = input[medians[col] + col*rows];
}

__device__ float hermite_polynomial(unsigned int order, float x) {
        switch (order) {
                case 0:
                        return 1.0;

                case 1:
                        return 2.0*x;

                default:
                        return 2.0*x*hermite_polynomial(order-1, x)
                                -2.0*(order-1)*hermite_polynomial(order-2, x);
        }
}

__device__ unsigned int factorial(unsigned int n)
{
        return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

__device__ float hermite_function(unsigned int order, float x) {
        return hermite_polynomial(order, x) / (
                        sqrt(pow(2.0, (double) order) * factorial(order) * sqrt(M_PI))
                        * exp(x*x/2)
                );
}

__global__ void PFunctionalHermite_kernel(const float *input,
                float *output, unsigned int order, int center)
{
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
        if ((row-1) < center)
                z = row * stepsize_lower - 10;
        else
                z = (row-center) * stepsize_upper;

        // Fetch
        temp[row] = input[row + col*rows] * hermite_function(order, z);
        __syncthreads();

        // Scan
        scan_array(temp, row, rows, SUM);

        // Write back
        if (row == rows-1)
                output[col] = temp[rows + row];
}

//
// T functionals
//

void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{
        // Launch radon kernel
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                TFunctionalRadon_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *output, a);
                CUDAHelper::checkState();
        }
}

void TFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{
        // Launch prefix sum kernel
        CUDAHelper::GlobalMemory<float> *prescan = new CUDAHelper::GlobalMemory<float>(input->sizes());
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                prescan_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *prescan, NONE);
                CUDAHelper::checkState();
        }

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<int> *medians = new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(input->cols()), 0);
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                findWeighedMedian_kernel<<<blocks, threads, input->rows()*sizeof(float)>>>(*input, *prescan, *medians);
                CUDAHelper::checkState();
        }
        delete prescan;

        // Launch T1 kernel
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                TFunctional1_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *medians, *output, a);
                CUDAHelper::checkState();
        }
        delete medians;
}

void TFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a)
{
        // Launch prefix sum kernel
        CUDAHelper::GlobalMemory<float> *prescan = new CUDAHelper::GlobalMemory<float>(input->sizes());
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                prescan_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *prescan, NONE);
                CUDAHelper::checkState();
        }

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<int> *medians = new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(input->cols()), 0);
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                findWeighedMedian_kernel<<<blocks, threads, input->rows()*sizeof(float)>>>(*input, *prescan, *medians);
                CUDAHelper::checkState();
        }
        delete prescan;

        // Launch T2 kernel
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                TFunctional2_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *medians, *output, a);
                CUDAHelper::checkState();
        }
        delete medians;
}

void TFunctional345(const CUDAHelper::GlobalMemory<float> *input,
                const CUDAHelper::GlobalMemory<float> *precalc_real,
                const CUDAHelper::GlobalMemory<float> *precalc_imag,
                CUDAHelper::GlobalMemory<float> *output, int a)
{
        // Launch prefix sum kernel
        CUDAHelper::GlobalMemory<float> *prescan = new CUDAHelper::GlobalMemory<float>(input->sizes());
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                prescan_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *prescan, SQRT);
                CUDAHelper::checkState();
        }

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<int> *medians = new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(input->cols()), 0);
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                findWeighedMedian_kernel<<<blocks, threads, input->rows()*sizeof(float)>>>(*input, *prescan, *medians);
                CUDAHelper::checkState();
        }
        delete prescan;

        // Launch T345 kernel
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                TFunctional345_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *medians, *precalc_real, *precalc_imag, *output, a);
                CUDAHelper::checkState();
        }
        delete medians;
}


//
// P-functionals
//

void PFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output)
{
        // Launch P1 kernel
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                PFunctional1_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *output);
                CUDAHelper::checkState();
        }
}

void PFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output)
{
        // Launch prefix sum kernel
        CUDAHelper::GlobalMemory<float> *prescan = new CUDAHelper::GlobalMemory<float>(input->sizes());
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                prescan_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *prescan, NONE);
                CUDAHelper::checkState();
        }

        // Launch weighed median kernel
        CUDAHelper::GlobalMemory<int> *medians = new CUDAHelper::GlobalMemory<int>(CUDAHelper::size_1d(input->cols()), 0);
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                findWeighedMedian_kernel<<<blocks, threads, input->rows()*sizeof(float)>>>(*input, *prescan, *medians);
                CUDAHelper::checkState();
        }
        delete prescan;

        // Launch P2 kernel
        {
                dim3 threads(1, 1);
                dim3 blocks(input->cols(), 1);
                PFunctional2_kernel<<<blocks, threads>>>(*input, *medians, *output, input->rows());
                CUDAHelper::checkState();
        }
        delete medians;
}

void PFunctional3(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output)
{

}

void PFunctionalHermite(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, unsigned int order,
                int center)
{
        // Launch P1 kernel
        {
                dim3 threads(1, input->rows());
                dim3 blocks(input->cols(), 1);
                PFunctionalHermite_kernel<<<blocks, threads, 2*input->rows()*sizeof(float)>>>(*input, *output, order, center);
                CUDAHelper::checkState();
        }
}
