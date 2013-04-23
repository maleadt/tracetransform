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

__device__ float interpolate_kernel(const Eigen::Map<const Eigen::MatrixXf> &source, const Point<float>::type &p)
{
        // Get fractional and integral part of the coordinates
        const int x_int = (int) p.x();
        const int y_int = (int) p.y();
        const float x_fract = p.x() - x_int;
        const float y_fract = p.y() - y_int;

        return    source(y_int, x_int)     * (1-x_fract)*(1-y_fract)
                + source(y_int, x_int+1)   * x_fract*(1-y_fract)
                + source(y_int+1, x_int)   * (1-x_fract)*y_fract
                + source(y_int+1, x_int+1) * x_fract*y_fract;

}

__constant__ float _transform[4];

__global__ void rotate_kernel(const float* _input, float* _output,
                const int cols, const int rows)
{
        // Compute thread dimension
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        // Get Eigen matrices back
        Eigen::Map<const Eigen::MatrixXf> input(_input, rows, cols);
        Eigen::Map<Eigen::MatrixXf> output(_output, rows, cols);
        Eigen::Map<const Eigen::Matrix2f> transform(_transform);

        Point<float>::type origin(cols / 2.0, rows / 2.0);

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

        // Calculate transform matrix
        Eigen::Matrix2f transform_data;
        transform_data <<       std::cos(angle), -std::sin(angle),
                                std::sin(angle),  std::cos(angle);
        CUDAHelper::ConstantMemory<float> *transform = new CUDAHelper::ConstantMemory<float>(_transform, 4);
        transform->upload(transform_data.data());

        dim3 threads(blocksize, blocksize);
        dim3 blocks(rows/blocksize, cols/blocksize);
        rotate_kernel<<<blocks, threads>>>(*input, *output, rows, cols);

        chrono.stop();
        clog(trace) << "Rotation kernel took " << chrono.elapsed() << " ms." << std::endl;
        return output;
}
