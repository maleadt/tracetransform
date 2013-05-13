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

enum class PFunctional
{
        Hermite,
        P1,
        P2,
        P3
};

struct PFunctionalArguments
{
        PFunctionalArguments(boost::optional<unsigned int> _order = boost::none,
                        boost::optional<size_t> _center = boost::none)
                        : order(_order), center(_center)
        {
        }

        // Arguments for Hermite P-functional
        boost::optional<unsigned int> order;
        boost::optional<size_t> center;
};

struct PFunctionalWrapper
{
        PFunctionalWrapper()
        {
        }

        PFunctionalWrapper(const std::string &_name,
                        const PFunctional &_functional,
                        const PFunctionalArguments &_arguments =
                                        PFunctionalArguments())
                        : name(_name), functional(_functional), arguments(
                                        _arguments)
        {
        }

        std::string name;
        PFunctional functional;
        PFunctionalArguments arguments;
};

std::istream& operator>>(std::istream& in, PFunctionalWrapper& wrapper);


//
// Module definitions
//

CUDAHelper::GlobalMemory<float> *nearest_orthonormal_sinogram(
        const CUDAHelper::GlobalMemory<float>* input,
        size_t& new_center);

Eigen::VectorXf getCircusFunction(
        const Eigen::MatrixXf &input,
        const PFunctionalWrapper &pfunctional);

#endif
