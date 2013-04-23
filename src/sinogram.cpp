//
// Configuration
//

// Header include
#include "sinogram.hpp"

// Standard library
#include <cassert>
#include <cmath>

// Boost
#include <boost/optional.hpp>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "kernels/rotate.hpp"
#include "kernels/functionals.hpp"


//
// Module definitions
//

Eigen::MatrixXf getSinogram(
        const Eigen::MatrixXf &input,
        const TFunctionalWrapper &tfunctional)
{
        assert(input.rows() == input.cols());   // padded image!

        // Calculate and allocate the output matrix
        size_t a_steps = 360;
        size_t p_steps = (size_t) input.rows();
        Eigen::MatrixXf output(p_steps, a_steps);

        // Upload the input image
        CUDAHelper::GlobalMemory<float> *input_mem =
                        new CUDAHelper::GlobalMemory<float>(
                                        input.rows() * input.cols());
        input_mem->upload(input.data());

        // Process all angles
        for (size_t a = 0; a < a_steps; a++) {
                // Rotate the image
                CUDAHelper::GlobalMemory<float> *input_rotated_mem = rotate(
                                input_mem, -deg2rad(a),
                                input.rows(), input.cols());
                Eigen::MatrixXf input_rotated(input.rows(), input.cols());
                input_rotated_mem->download(input_rotated.data());

                // Process all projection bands
                CUDAHelper::GlobalMemory<float> *output_mem;
                switch (tfunctional.functional) {
                        case TFunctional::Radon:
                                output_mem = TFunctionalRadon(input_rotated_mem, input.rows(), input.cols());
                                break;
                        case TFunctional::T1:
                                output_mem = TFunctional1(input_rotated_mem, input.rows(), input.cols());
                                break;
                        case TFunctional::T2:
                                output_mem = TFunctional2(input_rotated_mem, input.rows(), input.cols());
                                break;
                        case TFunctional::T3:
                                output_mem = TFunctional3(input_rotated_mem, input.rows(), input.cols());
                                break;
                        case TFunctional::T4:
                                output_mem = TFunctional4(input_rotated_mem, input.rows(), input.cols());
                                break;
                        case TFunctional::T5:
                                output_mem = TFunctional5(input_rotated_mem, input.rows(), input.cols());
                                break;
                }
                output_mem->download(output.data());
        }

        return output;
}
