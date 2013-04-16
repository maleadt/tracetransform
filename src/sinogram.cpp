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
#include "kernels/hello.hpp"


//
// Module definitions
//

Eigen::MatrixXd getSinogram(
        const Eigen::MatrixXd &input,
        const double a_stepsize,
        const double p_stepsize,
        FunctionalWrapper *tfunctional)
{
        assert(a_stepsize > 0);
        assert(p_stepsize > 0);
        assert(input.rows() == input.cols());   // padded image!

        hello();

        // Get the image origin to rotate around
        Point origin((input.cols()-1)/2.0, (input.rows()-1)/2.0);

        // Calculate and allocate the output matrix
        size_t a_steps = (size_t) std::floor(360 / a_stepsize);
        size_t p_steps = (size_t) std::floor(input.rows() / p_stepsize);
        Eigen::MatrixXd output((int) p_steps, (int) a_steps);

        // Process all angles
        for (size_t a_step = 0; a_step < a_steps; a_step++) {
                // Rotate the image
                double a = a_step * a_stepsize;
                Eigen::MatrixXd input_rotated = rotate(input, origin, deg2rad(a));

                // Process all projection bands
                for (size_t p_step = 0; p_step < p_steps; p_step++) {
                        double p = p_stepsize * p_step;
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
