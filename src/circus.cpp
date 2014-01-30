//
// Configuration
//

// Header include
#include "circus.hpp"

// Standard library
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>

// Eigen
#include <Eigen/SVD>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "kernels/functionals.hpp"


//
// Module definitions
//

std::istream &operator>>(std::istream &in, PFunctionalWrapper &wrapper) {
    in >> wrapper.name;
    if (wrapper.name == "1") {
        wrapper.name = "P1";
        wrapper.functional = PFunctional::P1;
    } else if (wrapper.name == "2") {
        wrapper.name = "P2";
        wrapper.functional = PFunctional::P2;
    } else if (wrapper.name == "3") {
        wrapper.name = "P3";
        wrapper.functional = PFunctional::P3;
#ifdef WITH_CULA
    } else if (wrapper.name[0] == 'H') {
        wrapper.functional = PFunctional::Hermite;
        if (wrapper.name.size() < 2)
            throw boost::program_options::validation_error(
                boost::program_options::validation_error::invalid_option_value,
                "Missing order parameter for Hermite P-functional");
        try {
            wrapper.arguments.order =
                boost::lexical_cast<unsigned int>(wrapper.name.substr(1));
        }
        catch (boost::bad_lexical_cast &) {
            throw boost::program_options::validation_error(
                boost::program_options::validation_error::invalid_option_value,
                "Unparseable order parameter for Hermite P-functional");
        }
#endif
    } else {
        throw boost::program_options::validation_error(
            boost::program_options::validation_error::invalid_option_value,
            "Unknown P-functional");
    }
    return in;
}

std::vector<CUDAHelper::GlobalMemory<float> *>
getCircusFunctions(const CUDAHelper::GlobalMemory<float> *input,
                   const std::vector<PFunctionalWrapper> &pfunctionals) {
    // Allocate the output matrices
    std::vector<CUDAHelper::GlobalMemory<float> *> outputs(pfunctionals.size());
    for (size_t p = 0; p < pfunctionals.size(); p++)
        outputs[p] = new CUDAHelper::GlobalMemory<float>(
            CUDAHelper::size_1d(input->cols()));

    // Pre-calculate
    // NOTE: some of these pre-calculations are just memory allocations
    std::map<PFunctional, void *> precalculations;
    for (size_t p = 0; p < pfunctionals.size(); p++) {
        PFunctional pfunctional = pfunctionals[p].functional;
        switch (pfunctional) {
        case PFunctional::P2:
            precalculations[pfunctional] =
                PFunctional2_prepare(input->rows(), input->cols());
            break;
        case PFunctional::P1:
        case PFunctional::P3:
        default:
            break;
        }
    }

    // Process all P-functionals
    for (size_t p = 0; p < pfunctionals.size(); p++) {
        PFunctional pfunctional = pfunctionals[p].functional;
        switch (pfunctionals[p].functional) {
        case PFunctional::P1:
            PFunctional1(input, outputs[p]);
            break;
        case PFunctional::P2:
            PFunctional2(input,
                         (PFunctional2_precalc_t *)precalculations[pfunctional],
                         outputs[p]);
            break;
        case PFunctional::P3:
            PFunctional3(input, outputs[p]);
            break;
#ifdef WITH_CULA
        case PFunctional::Hermite:
            PFunctionalHermite(input, outputs[p], *pfunctional.arguments.order,
                               *pfunctional.arguments.center);
            break;
#endif
        }
    }

    // Destroy pre-calculations
    std::map<PFunctional, void *>::iterator it = precalculations.begin();
    while (it != precalculations.end()) {
        switch (it->first) {
        case PFunctional::P2: {
            PFunctional2_precalc_t *precalc =
                (PFunctional2_precalc_t *)it->second;
            PFunctional2_destroy(precalc);
            break;
        }
        case PFunctional::P1:
        case PFunctional::P3:
        default:
            break;
        }
        ++it;
    }

    return outputs;
}
