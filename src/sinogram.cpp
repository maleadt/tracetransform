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
        const TFunctionalWrapper &tfunctional)
{
        assert(input.rows() == input.cols());   // padded image!

        // Get the image origin to rotate around
        Point<float>::type origin((input.cols()-1)/2.0, (input.rows()-1)/2.0);

        // Calculate and allocate the output matrix
        int a_steps = 360;
        int p_steps = input.cols();
        Eigen::MatrixXf output(p_steps, a_steps);

        // Process all angles
        for (int a = 0; a < a_steps; a++) {
                // Rotate the image
                Eigen::MatrixXf input_rotated = rotate(input, origin, -deg2rad(a));

                // Process all projection bands
                for (int p = 0; p < p_steps; p++) {
                        float *data = input_rotated.data() + p * input.rows();
                        int length = input.rows();
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
                                p,      // row
                                a       // column
                        ) = result;
                }
        }

        return output;
}
