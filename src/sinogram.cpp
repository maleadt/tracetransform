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
        const float a_stepsize,
        const float p_stepsize,
        const TFunctionalWrapper &tfunctional)
{
        assert(a_stepsize > 0);
        assert(p_stepsize > 0);
        assert(input.rows() == input.cols());   // padded image!

        // Calculate and allocate the output matrix
        size_t a_steps = (size_t) std::floor(360 / a_stepsize);
        size_t p_steps = (size_t) std::floor(input.rows() / p_stepsize);
        Eigen::MatrixXf output((int) p_steps, (int) a_steps);

        // Upload the input image
        CUDAHelper::GlobalMemory<float> *input_mem =
                        new CUDAHelper::GlobalMemory<float>(
                                        input.rows() * input.cols());
        input_mem->upload(input.data());

        // Process all angles
        for (size_t a_step = 0; a_step < a_steps; a_step++) {
                // Rotate the image
                float a = a_step * a_stepsize;
                CUDAHelper::GlobalMemory<float> *input_rotated_mem = rotate(
                                input_mem, -deg2rad(a),
                                input.rows(), input.cols());
                Eigen::MatrixXf input_rotated(input.rows(), input.cols());
                input_rotated_mem->download(input_rotated.data());

                // Process all projection bands
                for (size_t p_step = 0; p_step < p_steps; p_step++) {
                        float p = p_stepsize * p_step;

                        float *data = input_rotated.data() + ((size_t) std::floor(p)) * input.rows();
                        size_t length = input.rows();
                        float result;
                        switch (tfunctional.functional) {
                                case TFunctional::Radon:
                                        result = TFunctionalRadon(data, length);
                                        break;
                                case TFunctional::T1:
                                        result = TFunctional1(data, length);
                                        break;
                                case TFunctional::T2:
                                        result = TFunctional2(data, length);
                                        break;
                                case TFunctional::T3:
                                        result = TFunctional3(data, length);
                                        break;
                                case TFunctional::T4:
                                        result = TFunctional4(data, length);
                                        break;
                                case TFunctional::T5:
                                        result = TFunctional5(data, length);
                                        break;
                        }
                        output(
                                p_step,        // row
                                a_step         // column
                        ) = result;
                }
        }

        return output;
}
