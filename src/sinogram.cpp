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
        const Eigen::MatrixXf &input, unsigned int angle_stepsize,
        const std::vector<TFunctionalWrapper> &tfunctionals)
{
        assert(input.rows() == input.cols());   // padded image!

        // Get the image origin to rotate around
        Point<float>::type origin((input.cols()-1)/2.0, (input.rows()-1)/2.0);

        // Calculate and allocate the output matrix
        int a_steps = (int) std::floor(360 / angle_stepsize);
        int p_steps = input.cols();
        std::vector<Eigen::MatrixXf> outputs(tfunctionals.size());
        for (size_t t = 0; t < tfunctionals.size(); t++)
                outputs[t] = Eigen::MatrixXf(p_steps, a_steps);

        // Pre-calculate
        int length = input.rows();
        TFunctional345Precalculation *t3precalc = 0, *t4precalc = 0, *t5precalc = 0;
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                switch (tfunctionals[t].functional) {
                        case TFunctional::T3:
                                if (t3precalc == 0) {
                                        t3precalc = new TFunctional345Precalculation(length);
                                        for (int r = 1; r < length; r++) {
                                                t3precalc->real[r] = r*cos(5.0*log(r));
                                                t3precalc->imag[r] = r*sin(5.0*log(r));
                                        }
                                }
                                break;
                        case TFunctional::T4:
                                if (t4precalc == 0) {
                                        t4precalc = new TFunctional345Precalculation(length);
                                        for (int r = 1; r < length; r++) {
                                        t4precalc->real[r] = cos(3.0*log(r));
                                        t4precalc->imag[r] = sin(3.0*log(r));
                                        }
                                }
                                break;
                        case TFunctional::T5:
                                if (t5precalc == 0) {
                                        t5precalc = new TFunctional345Precalculation(length);
                                        for (int r = 1; r < length; r++) {
                                        t5precalc->real[r] = sqrt(r)*cos(4.0*log(r));
                                        t5precalc->imag[r] = sqrt(r)*sin(4.0*log(r));
                                        }
                                }
                                break;
                }
        }

        // Process all angles
        for (int a_step = 0; a_step < a_steps; a_step++) {
                // Rotate the image
                float a = a_step * angle_stepsize;
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
                                                result = TFunctional345(data, t3precalc->real, t3precalc->imag, length);
                                                break;
                                        case TFunctional::T4:
                                                result = TFunctional345(data, t4precalc->real, t4precalc->imag, length);
                                                break;
                                        case TFunctional::T5:
                                                result = TFunctional345(data, t5precalc->real, t5precalc->imag, length);
                                                break;
                                }
                                outputs[t](
                                        p,      // row
                                        a_step  // column
                                ) = result;
                        }
                }
        }

        if (t3precalc != 0)
                delete t3precalc;
        if (t4precalc != 0)
                delete t4precalc;
        if (t5precalc != 0)
                delete t5precalc;
        return outputs;
}
