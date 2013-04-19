//
// Configuration
//

// Header include
#include "sinogram.hpp"

// Standard library
#include <cassert>
#include <cmath>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "wrapper.hpp"
#include "kernels/rotate.hpp"


//
// Module definitions
//

Eigen::MatrixXf getSinogram(
        const Eigen::MatrixXf &input,
        const float a_stepsize,
        const float p_stepsize,
        FunctionalWrapper *tfunctional)
{
        assert(a_stepsize > 0);
        assert(p_stepsize > 0);
        assert(input.rows() == input.cols());   // padded image!

        // Get the image origin to rotate around
        Point<float>::type origin((input.cols()-1)/2.0, (input.rows()-1)/2.0);

        // Calculate and allocate the output matrix
        size_t a_steps = (size_t) std::floor(360 / a_stepsize);
        size_t p_steps = (size_t) std::floor(input.rows() / p_stepsize);
        Eigen::MatrixXf output((int) p_steps, (int) a_steps);

        // Process all angles
        for (size_t a_step = 0; a_step < a_steps; a_step++) {
                // Rotate the image
                float a = a_step * a_stepsize;
                Eigen::MatrixXf input_rotated = rotate(input, origin, -deg2rad(a));
                pgmWrite("eigen.pgm", mat2gray(input_rotated));

                // Rotate the image, using CUDA
                CUDAHelper::GlobalMemory<float> *input_mem =
                                new CUDAHelper::GlobalMemory<float>(
                                                input.rows() * input.cols());
                input_mem->upload(input.data());
                CUDAHelper::GlobalMemory<float> *input_rotated_mem = rotate(
                                input_mem, -deg2rad(a),
                                input.rows(), input.cols());
                input_rotated_mem->download(input_rotated.data());
                pgmWrite("cuda.pgm", mat2gray(input_rotated));
                if (a_step == 10)
                        exit(0);

                // Process all projection bands
                for (size_t p_step = 0; p_step < p_steps; p_step++) {
                        float p = p_stepsize * p_step;
                        output(
                                p_step,        // row
                                a_step         // column
                        ) = (*tfunctional)(
                                input_rotated.data() + ((size_t) std::floor(p)) * input.rows(),
                                input.rows());
                }
        }

        return output;
}
