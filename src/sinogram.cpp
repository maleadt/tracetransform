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
        const CUDAHelper::GlobalMemory<float> *input,
        const std::vector<TFunctionalWrapper> &tfunctionals)
{
        assert(input->size(0) == input->size(1));   // padded image!
        const int rows = input->size(0);
        const int cols = input->size(1);

        // Calculate and allocate the output matrix
        int a_steps = 360;
        int p_steps = cols;
        std::vector<CUDAHelper::GlobalMemory<float>*> outputs(tfunctionals.size());
        for (size_t t = 0; t < tfunctionals.size(); t++)
                outputs[t] = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(p_steps, a_steps), 0);

        // Allocate intermediary matrix for rotated image
        CUDAHelper::GlobalMemory<float> *input_rotated = new CUDAHelper::GlobalMemory<float>(input->sizes());

        // Process all angles
        for (int a = 0; a < a_steps; a++) {
                // Rotate the image
                rotate(input, input_rotated, -deg2rad(a));

                // Process all T-functionals
                for (size_t t = 0; t < tfunctionals.size(); t++) {
                        // Process all projection bands
                        switch (tfunctionals[t].functional) {
                                case TFunctional::Radon:
                                        TFunctionalRadon(input_rotated, outputs[t], a);
                                        break;
                                case TFunctional::T1:
                                        TFunctional1(input_rotated, outputs[t], a);
                                        break;
                                case TFunctional::T2:
                                        TFunctional2(input_rotated, outputs[t], a);
                                        break;
                                case TFunctional::T3:
                                        TFunctional3(input_rotated, outputs[t], a);
                                        break;
                                case TFunctional::T4:
                                        TFunctional4(input_rotated, outputs[t], a);
                                        break;
                                case TFunctional::T5:
                                        TFunctional5(input_rotated, outputs[t], a);
                                        break;
                        }
                }
        }

        delete input_rotated;
        return outputs;
}
