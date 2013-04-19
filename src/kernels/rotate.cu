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

// Static parameters
const int blocksize = 16;


//
// Kernels
//

typedef Eigen::RowVector2i Pointi;
typedef Eigen::RowVector2f Pointf;

__global__ void linear_rotate_kernel(const float* input, float* output, int width,
                int height, float angle)
{
        Pointi p(blockIdx.x * blockDim.x + threadIdx.x,
                        blockIdx.y * blockDim.y + threadIdx.y);
        Pointf origin(width / 2, height / 2);
        Pointi q(
                        cos(angle) * (p.x() - origin.x())
                                        - sin(angle) * (p.y() - origin.y())
                                        + origin.x(),
                        sin(angle) * (p.x() - origin.x())
                                        + cos(angle) * (p.y() - origin.y())
                                        + origin.y());
        if (q.x() >= 0 && q.x() < width && q.y() >= 0 && q.y() < height)
                output[p.x() + p.y() * width] = input[q.x() + q.y() * width];
}

__global__ void bilinear_rotate_kernel(const float* input, float* output,
                const unsigned int width, const unsigned int height,
                const float angle)
{
        // compute thread dimension
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        // compute target address
        const unsigned int idx = x + y * width;

        const int xA = (x - width / 2);
        const int yA = (y - height / 2);

        const int xR = (int) (xA * cos(angle) - yA * sin(angle));
        const int yR = (int) (xA * sin(angle) + yA * cos(angle));

        float src_x = xR + width / 2;
        float src_y = yR + height / 2;

        if (src_x >= 0.0f && src_x < width && src_y >= 0.0f && src_y < height) {
                // BI - LINEAR INTERPOLATION
                float src_x0 = (float) (int) (src_x);
                float src_x1 = (src_x0 + 1);
                float src_y0 = (float) (int) (src_y);
                float src_y1 = (src_y0 + 1);

                float sx = (src_x - src_x0);
                float sy = (src_y - src_y0);

                int idx_src00 = min(max(0.0f, src_x0 + src_y0 * width),
                                width * height - 1.0f);
                int idx_src10 = min(max(0.0f, src_x1 + src_y0 * width),
                                width * height - 1.0f);
                int idx_src01 = min(max(0.0f, src_x0 + src_y1 * width),
                                width * height - 1.0f);
                int idx_src11 = min(max(0.0f, src_x1 + src_y1 * width),
                                width * height - 1.0f);

                output[idx] = (1.0f - sx) * (1.0f - sy) * input[idx_src00];
                output[idx] += (sx) * (1.0f - sy) * input[idx_src10];
                output[idx] += (1.0f - sx) * (sy) * input[idx_src01];
                output[idx] += (sx) * (sy) * input[idx_src11];
        } else {
                output[idx] = 0.0f;
        }
}


//
// Wrappers
//

CUDAHelper::GlobalMemory<float> *rotate(
                const CUDAHelper::GlobalMemory<float> *input, float angle,
                int rows, int cols)
{
        assert(rows*cols == input->size());

        CUDAHelper::GlobalMemory<float> *output = new CUDAHelper::GlobalMemory<float>(input->size(), 0);

        dim3 threads(blocksize, blocksize);
        dim3 blocks(rows/blocksize, cols/blocksize);
        linear_rotate_kernel<<<blocks, threads>>>(*input, *output, rows, cols, angle);

        return output;
}
