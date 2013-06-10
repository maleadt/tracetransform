//
// Configuration
//

// Header include
#include "sinogram.hpp"

// Standard library
#include <cassert>
#include <cmath>
#include <complex>

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

std::istream& operator>>(std::istream& in, TFunctionalWrapper& wrapper)
{
        in >> wrapper.name;
        if (wrapper.name == "0") {
                wrapper.name = "Radon";
                wrapper.functional = TFunctional::Radon;
        } else if (wrapper.name == "1") {
                wrapper.name = "T1";
                wrapper.functional = TFunctional::T1;
        } else if (wrapper.name == "2") {
                wrapper.name = "T2";
                wrapper.functional = TFunctional::T2;
        } else if (wrapper.name == "3") {
                wrapper.name = "T3";
                wrapper.functional = TFunctional::T3;
        } else if (wrapper.name == "4") {
                wrapper.name = "T4";
                wrapper.functional = TFunctional::T4;
        } else if (wrapper.name == "5") {
                wrapper.name = "T5";
                wrapper.functional = TFunctional::T5;
        } else {
                throw boost::program_options::validation_error(
                        boost::program_options::validation_error::invalid_option_value,
                        "Unknown T-functional");
        }
        return in;
}

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

        // Pre-calculate
        int length = input.rows();
        float *precalc_real = new float[length];
        float *precalc_imag = new float[length];
        switch (tfunctional.functional) {
                case TFunctional::T3:
                {
                        for (int r = 1; r < length; r++) {
                                precalc_real[r] = r*cos(5.0*log(r));
                                precalc_imag[r] = r*sin(5.0*log(r));
                        }
                        break;
                }
                case TFunctional::T4:
                {
                        for (int r = 1; r < length; r++) {
                                precalc_real[r] = cos(3.0*log(r));
                                precalc_imag[r] = sin(3.0*log(r));
                        }
                        break;
                }
                case TFunctional::T5:
                {
                        for (int r = 1; r < length; r++) {
                                precalc_real[r] = sqrt(r)*cos(4.0*log(r));
                                precalc_imag[r] = sqrt(r)*sin(4.0*log(r));
                        }
                        break;
                }
        }

        // Process all angles
        #pragma omp parallel for
        for (int a = 0; a < a_steps; a++) {
                // Rotate the image
                Eigen::MatrixXf input_rotated = rotate(input, origin, -deg2rad(a));

                // Process all projection bands
                for (int p = 0; p < p_steps; p++) {
                        float *data = input_rotated.data() + p * input.rows();
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
                                case TFunctional::T4:
                                case TFunctional::T5:
                                        result = TFunctional345(data, precalc_real, precalc_imag, length);
                                        break;
                        }
                        output(
                                p,      // row
                                a       // column
                        ) = result;
                }
        }

        delete[] precalc_real, precalc_real;
        return output;
}
