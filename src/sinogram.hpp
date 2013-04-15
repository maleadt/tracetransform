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
