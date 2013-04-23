//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_SINOGRAM_
#define _TRACETRANSFORM_SINOGRAM_

// Standard library
#include <cstddef>

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


//
// Module definitions
//

Eigen::MatrixXf getSinogram(
        const Eigen::MatrixXf &input,
        const float a_stepsize,
        const float p_stepsize,
        const TFunctionalWrapper &tfunctional);

#endif
