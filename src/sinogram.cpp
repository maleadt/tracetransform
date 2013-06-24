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
#include "cudahelper/memory.hpp"
#include "kernels/rotate.hpp"
#include "kernels/functionals.hpp"


//
// Structures
//

struct TFunctional345Precalculation {
        TFunctional345Precalculation(size_t length)
        {
                real = new float[length];
                imag = new float[length];

                real_mem = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(length));
                imag_mem = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(length));
        }

        ~TFunctional345Precalculation()
        {
                delete[] real;
                delete[] imag;

                delete real_mem;
                delete imag_mem;
        }

        void upload ()
        {
                real_mem->upload(real);
                imag_mem->upload(imag);
        }

        float *real;
        float *imag;

        CUDAHelper::GlobalMemory<float> *real_mem;
        CUDAHelper::GlobalMemory<float> *imag_mem;
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

std::vector<CUDAHelper::GlobalMemory<float>*> getSinograms(
        const CUDAHelper::GlobalMemory<float> *input, unsigned int angle_stepsize,
        const std::vector<TFunctionalWrapper> &tfunctionals)
{
        assert(input->size(0) == input->size(1)); // padded image!
        const int rows = input->size(0);
        const int cols = input->size(1);

        // Calculate and allocate the output matrix
        int a_steps = (int) std::floor(360 / angle_stepsize);
        int p_steps = cols;
        std::vector<CUDAHelper::GlobalMemory<float>*> outputs(tfunctionals.size());
        for (size_t t = 0; t < tfunctionals.size(); t++)
                outputs[t] = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(p_steps, a_steps), 0);

        // Pre-calculate
        int length = rows;
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

                        t345precalc->upload();
                }
        }

        // Allocate intermediary matrix for rotated image
        CUDAHelper::GlobalMemory<float> *input_rotated = new CUDAHelper::GlobalMemory<float>(input->sizes());

        // Process all angles
        for (int a_step = 0; a_step < a_steps; a_step++) {
                // Rotate the image
                float a = a_step * angle_stepsize;
                rotate(input, input_rotated, -deg2rad(a));

                // Process all T-functionals
                for (size_t t = 0; t < tfunctionals.size(); t++) {
                        // Process all projection bands
                        switch (tfunctionals[t].functional) {
                                case TFunctional::Radon:
                                        TFunctionalRadon(input_rotated, outputs[t], a_step);
                                        break;
                                case TFunctional::T1:
                                        TFunctional1(input_rotated, outputs[t], a_step);
                                        break;
                                case TFunctional::T2:
                                        TFunctional2(input_rotated, outputs[t], a_step);
                                        break;
                                case TFunctional::T3:
                                case TFunctional::T4:
                                case TFunctional::T5:
                                        TFunctional345(input_rotated, t345precalc->real_mem, t345precalc->imag_mem, outputs[t], a_step);
                                        break;
                        }
                }
        }

        delete input_rotated;
        if (t345precalc != 0)
                delete t345precalc;
        return outputs;
}
