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
            boost::program_options::validation_error::invalid_option_value,
            "Unknown T-functional");
    }
    return in;
}

std::vector<CUDAHelper::GlobalMemory<float> *>
getSinograms(const CUDAHelper::GlobalMemory<float> *input,
             unsigned int angle_stepsize,
             const std::vector<TFunctionalWrapper> &tfunctionals) {
    assert(input->size(0) == input->size(1)); // padded image!
    //const int rows = input->size(0);
    const int cols = input->size(1);

    // Calculate and allocate the output matrices
    int a_steps = (int)std::floor(360 / angle_stepsize);
    int p_steps = cols;
    std::vector<CUDAHelper::GlobalMemory<float> *> outputs(tfunctionals.size());
    for (size_t t = 0; t < tfunctionals.size(); t++)
        outputs[t] = new CUDAHelper::GlobalMemory<float>(
            CUDAHelper::size_2d(p_steps, a_steps), 0);

    // Pre-calculate
    // NOTE: some of these pre-calculations are just memory allocations
    std::map<TFunctional, void *> precalculations;
    for (size_t t = 0; t < tfunctionals.size(); t++) {
        TFunctional tfunctional = tfunctionals[t].functional;
        switch (tfunctional) {
        case TFunctional::T1:
        case TFunctional::T2:
            precalculations[tfunctional] =
                TFunctional12_prepare(input->rows(), input->cols());
            break;
        case TFunctional::T3:
            precalculations[tfunctional] =
                TFunctional3_prepare(input->rows(), input->cols());
            break;
        case TFunctional::T4:
            precalculations[tfunctional] =
                TFunctional4_prepare(input->rows(), input->cols());
            break;
        case TFunctional::T5:
            precalculations[tfunctional] =
                TFunctional5_prepare(input->rows(), input->cols());
            break;
        case TFunctional::T6:
            precalculations[tfunctional] =
                TFunctional6_prepare(input->rows(), input->cols());
            break;
        case TFunctional::T7:
            precalculations[tfunctional] =
                TFunctional7_prepare(input->rows(), input->cols());
            break;
        case TFunctional::Radon:
        default:
            break;
        }
    }

    // Allocate intermediary matrix for rotated image
    CUDAHelper::GlobalMemory<float> *input_rotated =
        new CUDAHelper::GlobalMemory<float>(input->sizes());

    // Process all angles
    for (int a_step = 0; a_step < a_steps; a_step++) {
        // Rotate the image
        float a = a_step * angle_stepsize;
        rotate(input, input_rotated, -deg2rad(a));

        // Process all T-functionals
        for (size_t t = 0; t < tfunctionals.size(); t++) {
            TFunctional tfunctional = tfunctionals[t].functional;
            switch (tfunctionals[t].functional) {
            case TFunctional::Radon:
                TFunctionalRadon(input_rotated, outputs[t], a_step);
                break;
            case TFunctional::T1:
                TFunctional1(
                    input_rotated,
                    (TFunctional12_precalc_t *)precalculations[tfunctional],
                    outputs[t], a_step);
                break;
            case TFunctional::T2:
                TFunctional2(
                    input_rotated,
                    (TFunctional12_precalc_t *)precalculations[tfunctional],
                    outputs[t], a_step);
                break;
            case TFunctional::T3:
            case TFunctional::T4:
            case TFunctional::T5:
                TFunctional345(
                    input_rotated,
                    (TFunctional345_precalc_t *)precalculations[tfunctional],
                    outputs[t], a_step);
                break;
            case TFunctional::T6:
                TFunctional6(
                    input_rotated,
                    (TFunctional6_precalc_t *)precalculations[tfunctional],
                    outputs[t], a_step);
            	break;
            case TFunctional::T7:
                TFunctional7(
                    input_rotated,
                    (TFunctional7_precalc_t *)precalculations[tfunctional],
                    outputs[t], a_step);
                break;
            }
        }
    }

    delete input_rotated;

    // Destroy pre-calculations
    std::map<TFunctional, void *>::iterator it = precalculations.begin();
    while (it != precalculations.end()) {
        switch (it->first) {
        case TFunctional::T1:
        case TFunctional::T2: {
            TFunctional12_precalc_t *precalc =
                (TFunctional12_precalc_t *)it->second;
            TFunctional12_destroy(precalc);
            break;
        }
        case TFunctional::T3:
        case TFunctional::T4:
        case TFunctional::T5: {
            TFunctional345_precalc_t *precalc =
                (TFunctional345_precalc_t *)it->second;
            TFunctional345_destroy(precalc);
            break;
        }
        case TFunctional::T6: {
            TFunctional6_precalc_t *precalc =
                (TFunctional6_precalc_t *)it->second;
            TFunctional6_destroy(precalc);
            break;
        }
        case TFunctional::T7: {
            TFunctional7_precalc_t *precalc =
                (TFunctional7_precalc_t *)it->second;
            TFunctional7_destroy(precalc);
            break;
        }
        case TFunctional::Radon:
        default:
            break;
        }
        ++it;
    }

    return outputs;
}
