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
// Structures
//

struct TFunctional345Precalculation {
        TFunctional345Precalculation(size_t length)
        {
                real = new float[length];
                imag = new float[length];
        }

        ~TFunctional345Precalculation()
        {
                delete[] real;
                delete[] imag;
        }

        float *real;
        float *imag;
};


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

std::vector<Eigen::MatrixXf> getSinograms(
        const Eigen::MatrixXf &input,
        const std::vector<TFunctionalWrapper> &tfunctionals)
{
        assert(input.rows() == input.cols());   // padded image!

        // Get the image origin to rotate around
        Point<float>::type origin((input.cols()-1)/2.0, (input.rows()-1)/2.0);

        // Calculate and allocate the output matrix
        int a_steps = 360;
        int p_steps = input.cols();
        std::vector<Eigen::MatrixXf> outputs(tfunctionals.size());
        for (size_t t = 0; t < tfunctionals.size(); t++)
                outputs[t] = Eigen::MatrixXf(p_steps, a_steps);

        // Pre-calculate
        int length = input.rows();
        TFunctional345Precalculation *t345precalc = 0;
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                if (t345precalc == 0 && (
                                tfunctionals[t].functional == TFunctional::T3 ||
                                tfunctionals[t].functional == TFunctional::T4 ||
                                tfunctionals[t].functional == TFunctional::T5)) {
                        t345precalc = new TFunctional345Precalculation(length);

                        if (tfunctionals[t].functional == TFunctional::T3) {
                                for (int r = 1; r < length; r++) {
                                        t345precalc->real[r] = r*cos(5.0*log(r));
                                        t345precalc->imag[r] = r*sin(5.0*log(r));
                                }
                        } else if (tfunctionals[t].functional == TFunctional::T4) {
                                for (int r = 1; r < length; r++) {
                                        t345precalc->real[r] = cos(3.0*log(r));
                                        t345precalc->imag[r] = sin(3.0*log(r));
                                }
                        } else if (tfunctionals[t].functional == TFunctional::T5) {
                                for (int r = 1; r < length; r++) {
                                        t345precalc->real[r] = sqrt(r)*cos(4.0*log(r));
                                        t345precalc->imag[r] = sqrt(r)*sin(4.0*log(r));
                                }
                        }
                }
        }

        // Process all angles
        for (int a = 0; a < a_steps; a++) {
                // Rotate the image
                Eigen::MatrixXf input_rotated = rotate(input, origin, -deg2rad(a));

                // Process all projection bands
                for (int p = 0; p < p_steps; p++) {
                        float *data = input_rotated.data() + p * input.rows();

                        // Process all T-functionals
                        for (size_t t = 0; t < tfunctionals.size(); t++) {
                                float result;
                                switch (tfunctionals[t].functional) {
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
                                                result = TFunctional345(data, t345precalc->real, t345precalc->imag, length);
                                                break;
                                }
                                outputs[t](
                                        p,      // row
                                        a       // column
                                ) = result;
                        }
                }
        }

        if (t345precalc != 0)
                delete t345precalc;
        return outputs;
}
