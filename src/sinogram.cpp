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

        // Process all angles
        for (int a = 0; a < a_steps; a++) {
                // Rotate the image
                Eigen::MatrixXf input_rotated = rotate(input, origin, -deg2rad(a));

                // Process all projection bands
                for (int p = 0; p < p_steps; p++) {
                        float *data = input_rotated.data() + p * input.rows();
                        int length = input.rows();
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
                                                result = TFunctional3(data, length);
                                                break;
                                        case TFunctional::T4:
                                                result = TFunctional4(data, length);
                                                break;
                                        case TFunctional::T5:
                                                result = TFunctional5(data, length);
                                                break;
                                        default:
                                                assert(false);
                                }
                                outputs[t](
                                        p,      // row
                                        a       // column
                                ) = result;
                        }
                }
        }

        return outputs;
}
