//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_SINOGRAM_
#define _TRACETRANSFORM_SINOGRAM_

// Standard library
#include <cstddef>
#include <vector>

// Boost
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Dense>


//
// Functionals
//

enum class TFunctional
{
        Radon,
        T1,
        T2,
        T3,
        T4,
        T5
};

struct TFunctionalArguments
{
};

struct TFunctionalWrapper
{
        TFunctionalWrapper()
        {
        }

        TFunctionalWrapper(const std::string &_name,
                        const TFunctional &_functional,
                        const TFunctionalArguments &_arguments =
                                        TFunctionalArguments())
                        : name(_name), functional(_functional), arguments(
                                        _arguments)
        {
        }

        std::string name;
        TFunctional functional;
        TFunctionalArguments arguments;
};

std::istream& operator>>(std::istream& in, TFunctionalWrapper& wrapper);


//
// Module definitions
//

std::vector<Eigen::MatrixXf> getSinograms(
        const Eigen::MatrixXf &input, unsigned int angle_stepsize,
        const std::vector<TFunctionalWrapper> &tfunctionals);

#endif
