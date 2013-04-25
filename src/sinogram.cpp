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
#include "cudahelper/memory.hpp"
#include "kernels/rotate.hpp"
#include "kernels/functionals.hpp"


//
// Module definitions
//

// TODO: allow processing of multiple functionals, rotating the image only once
CUDAHelper::GlobalMemory<float> *getSinogram(
        const CUDAHelper::GlobalMemory<float> *input, const int rows, const int cols,
        const TFunctionalWrapper &tfunctional, int &output_rows, int &output_cols)
{
        assert(rows == cols);   // padded image!
        assert(input->size() == rows*cols);

        // Calculate and allocate the output matrix
        int a_steps = 360;
        int p_steps = cols;
        CUDAHelper::GlobalMemory<float> *output = new CUDAHelper::GlobalMemory<float>(p_steps * a_steps, 0);

        // Process all angles
        for (int a = 0; a < a_steps; a++) {
                // Rotate the image
                CUDAHelper::GlobalMemory<float> *input_rotated = rotate(
                                input, -deg2rad(a),
                                rows, cols);

                // Process all projection bands
                switch (tfunctional.functional) {
                        case TFunctional::Radon:
                                TFunctionalRadon(input_rotated, rows, cols, output, a);
                                break;
                        case TFunctional::T1:
                                TFunctional1(input_rotated, rows, cols, output, a);
                                break;
                        case TFunctional::T2:
                                TFunctional2(input_rotated, rows, cols, output, a);
                                break;
                        case TFunctional::T3:
                                TFunctional3(input_rotated, rows, cols, output, a);
                                break;
                        case TFunctional::T4:
                                TFunctional4(input_rotated, rows, cols, output, a);
                                break;
                        case TFunctional::T5:
                                TFunctional5(input_rotated, rows, cols, output, a);
                                break;
                }
                delete input_rotated;
        }

        output_rows = p_steps;
        output_cols = a_steps;
        return output;
}
