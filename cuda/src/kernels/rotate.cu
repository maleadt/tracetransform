//
// Configuration
//

// Header
#include "rotate.hpp"

// Standard library
#include <iostream>

// Local
#include "../logger.hpp"


//
// Kernels
//

__device__ float interpolate_kernel(const float *source, const int rows,
                                    const float x, const float y) {
    // Get fractional and integral part of the coordinates
    const int x_int = (int)x;
    const int y_int = (int)y;
    const float x_fract = x - x_int;
    const float y_fract = y - y_int;

    return source[y_int + x_int * rows] * (1 - x_fract) * (1 - y_fract) +
           source[y_int + (x_int + 1) * rows] * x_fract * (1 - y_fract) +
           source[y_int + 1 + x_int * rows] * (1 - x_fract) * y_fract +
           source[y_int + 1 + (x_int + 1) * rows] * x_fract * y_fract;
}

__global__ void rotate_kernel(const float *input, float *output, float angle) {
    // Compute the thread dimensions
    const int col = blockIdx.x;
    const int cols = gridDim.x;
    const int row = threadIdx.y;
    const int rows = blockDim.y;

    // Calculate transform matrix values
    float angle_cos = std::cos(angle);
    float angle_sin = std::sin(angle);

    // Calculate origin
    float xo = cols / 2.0;
    float yo = rows / 2.0;

    // Calculate the source location
    float xt = col - xo;
    float yt = row - yo;
    float x = xt * angle_cos + yt * angle_sin + xo;
    float y = -xt * angle_sin + yt * angle_cos + yo;

    // Interpolate the source value
    if (x >= 0 && x < cols - 1 && y >= 0 && y < rows - 1)
        output[row + col * rows] = interpolate_kernel(input, rows, x, y);
    else if (col < cols && row < rows)
        output[row + col * rows] = 0;
}


//
// Wrappers
//

void rotate(const CUDAHelper::GlobalMemory<float> *input,
            CUDAHelper::GlobalMemory<float> *output, float angle) {
    // Launch
    dim3 threads(1, input->rows());
    dim3 blocks(input->cols(), 1);
    rotate_kernel <<<blocks, threads>>> (*input, *output, angle);
    CUDAHelper::checkState();
}
