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
        CUDAHelper::GlobalMemory<float> *output = new CUDAHelper::GlobalMemory<float>(p_steps * a_steps, 0);

        // Upload the input image
        CUDAHelper::GlobalMemory<float> *input_mem =
                        new CUDAHelper::GlobalMemory<float>(
                                        input.rows() * input.cols());
        input_mem->upload(input.data());

        // Process all angles
        for (size_t a = 0; a < a_steps; a++) {
                // Rotate the image
                CUDAHelper::GlobalMemory<float> *input_rotated = rotate(
                                input_mem, -deg2rad(a),
                                input.rows(), input.cols());

                // Process all projection bands
                switch (tfunctional.functional) {
                        case TFunctional::Radon:
                                TFunctionalRadon(input_rotated, input.rows(), input.cols(), output, a);
                                break;
                        case TFunctional::T1:
                                TFunctional1(input_rotated, input.rows(), input.cols(), output, a);
                                break;
                        case TFunctional::T2:
                                TFunctional2(input_rotated, input.rows(), input.cols(), output, a);
                                break;
                        case TFunctional::T3:
                                TFunctional3(input_rotated, input.rows(), input.cols(), output, a);
                                break;
                        case TFunctional::T4:
                                TFunctional4(input_rotated, input.rows(), input.cols(), output, a);
                                break;
                        case TFunctional::T5:
                                TFunctional5(input_rotated, input.rows(), input.cols(), output, a);
                                break;
                }
        }

        Eigen::MatrixXf output_data(p_steps, a_steps);
        output->download(output_data.data());
        return output_data;
}
