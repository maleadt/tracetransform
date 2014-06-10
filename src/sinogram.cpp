//
// Configuration
//

// Header include
#include "sinogram.hpp"

// Standard library
#include <cassert> // for assert
#include <cmath>   // for floor
#include <cstddef> // for size_t
#include <map>     // for map, _Rb_tree_iterator, etc
#include <new>     // for operator new
#include <utility> // for pair

// Boost
#include <boost/program_options.hpp>

// Local
#include "global.hpp"
#include "auxiliary.hpp"
#include "functionals.hpp"


//
// Structures
//


//
// Module definitions
//

std::istream &operator>>(std::istream &in, TFunctionalWrapper &wrapper) {
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
    } else if (wrapper.name == "6") {
        wrapper.name = "T6";
        wrapper.functional = TFunctional::T6;
    } else if (wrapper.name == "7") {
        wrapper.name = "T7";
        wrapper.functional = TFunctional::T7;
    } else {
        throw boost::program_options::validation_error(
            boost::program_options::validation_error::invalid_option_value);
    }
    return in;
}

std::vector<Eigen::MatrixXf>
getSinograms(const Eigen::MatrixXf &input, unsigned int angle_stepsize,
             const std::vector<TFunctionalWrapper> &tfunctionals) {
    assert(input.rows() == input.cols()); // padded image!

    // Get the image origin to rotate around
    Point<float>::type origin((input.cols() - 1) / 2.0,
                              (input.rows() - 1) / 2.0);

    // Calculate and allocate the output matrices
    int a_steps = (int)std::floor(360 / angle_stepsize);
    std::vector<Eigen::MatrixXf> outputs(tfunctionals.size());
    for (size_t t = 0; t < tfunctionals.size(); t++)
        outputs[t] = Eigen::MatrixXf(input.cols(), a_steps);

    // Pre-calculate
    std::map<size_t, void *> precalculations;
    for (size_t t = 0; t < tfunctionals.size(); t++) {
        TFunctional tfunctional = tfunctionals[t].functional;
        switch (tfunctional) {
        case TFunctional::T3:
            precalculations[t] =
                TFunctional3_prepare(input.rows(), input.cols());
            break;
        case TFunctional::T4:
            precalculations[t] =
                TFunctional4_prepare(input.rows(), input.cols());
            break;
        case TFunctional::T5:
            precalculations[t] =
                TFunctional5_prepare(input.rows(), input.cols());
            break;
        case TFunctional::Radon:
        case TFunctional::T1:
        case TFunctional::T2:
        case TFunctional::T6:
        case TFunctional::T7:
        default:
            break;
        }
    }

    // Process all angles
    for (int a_step = 0; a_step < a_steps; a_step++) {
        // Rotate the image
        float a = a_step * angle_stepsize;
        Eigen::MatrixXf input_rotated = rotate(input, origin, deg2rad(a));

        // Process all projection bands
        for (int column = 0; column < input.cols(); column++) {
            Eigen::VectorXf data = input_rotated.col(column);

            // Process all T-functionals
            for (size_t t = 0; t < tfunctionals.size(); t++) {
                TFunctional tfunctional = tfunctionals[t].functional;
                float result;
                switch (tfunctional) {
                case TFunctional::Radon:
                    result = TFunctionalRadon(data);
                    break;
                case TFunctional::T1:
                    result = TFunctional1(data);
                    break;
                case TFunctional::T2:
                    result = TFunctional2(data);
                    break;
                case TFunctional::T3:
                case TFunctional::T4:
                case TFunctional::T5:
                    result = TFunctional345(data,
                        (TFunctional345_precalc_t *)precalculations[t]);
                    break;
                case TFunctional::T6:
                    result = TFunctional6(data);
                    break;
                case TFunctional::T7:
                    result = TFunctional7(data);
                    break;
                }
                outputs[t](column, // row (in the sinogram)
                           a_step  // column
                           ) = result;
            }
        }
    }

    // Destroy pre-calculations
    std::map<size_t, void *>::iterator it = precalculations.begin();
    while (it != precalculations.end()) {
        TFunctional tfunctional = tfunctionals[it->first].functional;
        switch (tfunctional) {
        case TFunctional::T3:
        case TFunctional::T4:
        case TFunctional::T5: {
            TFunctional345_precalc_t *precalc =
                (TFunctional345_precalc_t *)it->second;
            TFunctional345_destroy(precalc);
            break;
        }
        case TFunctional::Radon:
        case TFunctional::T1:
        case TFunctional::T2:
        case TFunctional::T6:
        case TFunctional::T7:
        default:
            break;
        }
        ++it;
    }

    return outputs;
}
