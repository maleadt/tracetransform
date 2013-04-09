//
// Configuration
//

// Include guard
#ifndef TRACETRANSFORM_SINOGRAM_HPP
#define TRACETRANSFORM_SINOGRAM_HPP

// Standard library
#include <cstddef>

// Eigen
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
#include <Eigen/Dense>

// Local
#include "wrapper.hpp"


//
// Module definitions
//

Eigen::MatrixXd getSinogram(
        const Eigen::MatrixXd &input,
        const double a_stepsize,
        const double p_stepsize,
        FunctionalWrapper *tfunctional);

#endif
