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

std::istream& operator>>(std::istream& in, PFunctionalWrapper& wrapper)
{
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
        } else if (wrapper.name[0] == 'H') {
                wrapper.functional = PFunctional::Hermite;
                if (wrapper.name.size() < 2)
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Missing order parameter for Hermite P-functional");
                try {
                        wrapper.arguments.order = boost::lexical_cast<unsigned int>(wrapper.name.substr(1));
                }
                catch(boost::bad_lexical_cast &) {
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Unparseable order parameter for Hermite P-functional");
                }
        } else {
                throw boost::program_options::validation_error(
                        boost::program_options::validation_error::invalid_option_value,
                        "Unknown P-functional");
        }
    return in;
}

CUDAHelper::GlobalMemory<float> *getCircusFunction(
        const CUDAHelper::GlobalMemory<float> *input,
        const PFunctionalWrapper &pfunctional)
{
        const int rows = input->size(0);
        const int cols = input->size(1);

        // Allocate the output matrix
        CUDAHelper::GlobalMemory<float> *output = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_1d(cols));

        // Trace all columns
        switch (pfunctional.functional) {
                case PFunctional::P1:
                        PFunctional1(input, output);
                        break;
                case PFunctional::P2:
                        PFunctional2(input, output);
                        break;
                case PFunctional::P3:
                        PFunctional3(input, output);
                        break;
                case PFunctional::Hermite:
                        PFunctionalHermite(input, output, *pfunctional.arguments.order, *pfunctional.arguments.center);
                        break;
        }

        return output;
}
