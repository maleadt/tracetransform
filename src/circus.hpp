//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CIRCUS_
#define _TRACETRANSFORM_CIRCUS_

// Standard library
#include <cstddef>

// Boost
#include <boost/optional.hpp>
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Dense>

// Local
#include "cudahelper/memory.hpp"


//
// Functionals
//

// TODO: make this an enum class when ICC 14 is released
enum PFunctional {
    Hermite,
    P1,
    P2,
    P3
};

struct PFunctionalArguments {
    PFunctionalArguments(boost::optional<unsigned int> _order = boost::none,
                         boost::optional<size_t> _center = boost::none)
        : order(_order), center(_center) {}

    // Arguments for Hermite P-functional
    boost::optional<unsigned int> order;
    boost::optional<size_t> center;
};

struct PFunctionalWrapper {
    PFunctionalWrapper() {}

    PFunctionalWrapper(const std::string &_name, const PFunctional &_functional,
                       const PFunctionalArguments &_arguments =
                           PFunctionalArguments())
        : name(_name), functional(_functional), arguments(_arguments) {}

    std::string name;
    PFunctional functional;
    PFunctionalArguments arguments;
};

std::istream &operator>>(std::istream &in, PFunctionalWrapper &wrapper);


//
// Module definitions
//

CUDAHelper::GlobalMemory<float> *
getCircusFunction(const CUDAHelper::GlobalMemory<float> *input,
                  const PFunctionalWrapper &pfunctional);

#endif
