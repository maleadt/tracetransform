//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_SINOGRAM_
#define _TRACETRANSFORM_SINOGRAM_

// Standard library
#include <istream> // for istream
#include <string>  // for string
#include <vector>  // for vector

// Eigen
#include <Eigen/Dense>


//
// Functionals
//

// TODO: make this an enum class when ICC 14 is released
enum TFunctional {
    Radon,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7
};

struct TFunctionalArguments {};

struct TFunctionalWrapper {
    TFunctionalWrapper() : name("invalid"), functional(TFunctional()) {
        // Invalid constructor, only used by boost::program_options
    }

    TFunctionalWrapper(const std::string &_name, const TFunctional &_functional,
                       const TFunctionalArguments &_arguments =
                           TFunctionalArguments())
        : name(_name), functional(_functional), arguments(_arguments) {}

    std::string name;
    TFunctional functional;
    TFunctionalArguments arguments;
};

std::istream &operator>>(std::istream &in, TFunctionalWrapper &wrapper);


//
// Module definitions
//

std::vector<Eigen::MatrixXf>
getSinograms(const Eigen::MatrixXf &input, unsigned int angle_stepsize,
             const std::vector<TFunctionalWrapper> &tfunctionals);

#endif
