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
extern "C" {
        #include "functionals.h"
}


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

        // Get the image origin to rotate around
        Point<float>::type origin((input.cols()-1)/2.0, (input.rows()-1)/2.0);

        // Calculate and allocate the output matrix
        size_t a_steps = (size_t) std::floor(360 / a_stepsize);
        size_t p_steps = (size_t) std::floor(input.rows() / p_stepsize);
        Eigen::MatrixXf output(p_steps, a_steps);

        // Process all angles
        for (size_t a_step = 0; a_step < a_steps; a_step++) {
                // Rotate the image
                float a = a_step * a_stepsize;
                Eigen::MatrixXf input_rotated = rotate(input, origin, -deg2rad(a));

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
