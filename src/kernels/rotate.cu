//
// Configuration
//

// Header
#include "rotate.hpp"

// Standard library
#include <iostream>

// Local
#include "../global.hpp"
#include "../logger.hpp"
#include "../cudahelper/chrono.hpp"

// Static parameters
const int blocksize = 16;


//
// Kernels
//

__global__ void linear_rotate_kernel(const float* input, float* output, int width,
                int height, float angle)
{
        Point<int>::type p(blockIdx.x * blockDim.x + threadIdx.x,
                        blockIdx.y * blockDim.y + threadIdx.y);
        Point<float>::type origin(width / 2, height / 2);
        Point<int>::type q(
                        cos(angle) * (p.x() - origin.x())
                                        - sin(angle) * (p.y() - origin.y())
                                        + origin.x(),
                        sin(angle) * (p.x() - origin.x())
                                        + cos(angle) * (p.y() - origin.y())
                                        + origin.y());
        if (q.x() >= 0 && q.x() < width && q.y() >= 0 && q.y() < height)
                output[p.x() + p.y() * width] = input[q.x() + q.y() * width];
}

__device__ float interpolate_kernel(const Eigen::Map<const Eigen::MatrixXf> &source, const Point<float>::type &p)
{
        // Get fractional and integral part of the coordinates
        float x_int, y_int;
        float x_fract = std::modf(p.x(), &x_int);
        float y_fract = std::modf(p.y(), &y_int);

        return    source((size_t)y_int, (size_t)x_int)*(1-x_fract)*(1-y_fract)
                + source((size_t)y_int, (size_t)x_int+1)*x_fract*(1-y_fract)
                + source((size_t)y_int+1, (size_t)x_int)*(1-x_fract)*y_fract
                + source((size_t)y_int+1, (size_t)x_int+1)*x_fract*y_fract;

}

__global__ void bilinear_rotate_kernel(const float* _input, float* _output,
                const unsigned int width, const unsigned int height,
                const float angle)
{
        // compute thread dimension
        const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

        // Get Eigen matrices back
        Eigen::Map<const Eigen::MatrixXf> input(_input, height, width);
        Eigen::Map<Eigen::MatrixXf> output(_output, height, width);

        Point<float>::type origin(width / 2.0, height / 2.0);

        // Calculate transform matrix
        Eigen::Matrix2f transform;
        transform <<    std::cos(angle), -std::sin(angle),
                        std::sin(angle),  std::cos(angle);

        // Process this point
        Point<float>::type p(col, row);
        Point<float>::type q = ((p - origin) * transform) + origin;
        if (    q.x() >= 0 && q.x() < input.cols()-1
                && q.y() >= 0 && q.y() < input.rows()-1)
                output(row, col) = interpolate_kernel(input, q);
}


//
// Wrappers
//

CUDAHelper::GlobalMemory<float> *rotate(
                const CUDAHelper::GlobalMemory<float> *input, float angle,
                int rows, int cols)
{
        assert(rows*cols == input->size());

        CUDAHelper::Chrono chrono;
        chrono.start();
        // TODO: why is memset required? Fails otherwise on e.g. angle=50deg
        CUDAHelper::GlobalMemory<float> *output = new CUDAHelper::GlobalMemory<float>(input->size(), 0);

        dim3 threads(blocksize, blocksize);
        dim3 blocks(rows/blocksize, cols/blocksize);
        bilinear_rotate_kernel<<<blocks, threads>>>(*input, *output, rows, cols, angle);

        chrono.stop();
        clog(trace) << "Rotation kernel took " << chrono.elapsed() << " ms." << std::endl;
        return output;
}
