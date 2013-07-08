//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CIRCUS_
#define _TRACETRANSFORM_CIRCUS_

// Standard library
#include <cstddef>                      // for size_t
#include <iosfwd>                       // for istream
#include <string>                       // for string

// Boost
#include <boost/none.hpp>               // for none
#include <boost/optional.hpp>           // for optional

// Eigen
#include <Eigen/Dense>                  // for MatrixXf, VectorXf


//
// Functionals
//

// TODO: make this an enum class when ICC 14 is released
enum PFunctional
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

Eigen::MatrixXf nearest_orthonormal_sinogram(
        const Eigen::MatrixXf &input,
        size_t& new_center);

Eigen::VectorXf getCircusFunction(
        const Eigen::MatrixXf &input,
        const PFunctionalWrapper &pfunctional);

#endif
