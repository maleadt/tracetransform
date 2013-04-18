//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CIRCUS_
#define _TRACETRANSFORM_CIRCUS_

// Standard library
#include <cstddef>

// Eigen
#include <Eigen/Dense>

// Local
#include "wrapper.hpp"


//
// Module definitions
//

Eigen::MatrixXf nearest_orthonormal_sinogram(
        const Eigen::MatrixXf &input,
        size_t& new_center);

Eigen::VectorXf getCircusFunction(
        const Eigen::MatrixXf &input,
        FunctionalWrapper *pfunctional);

#endif
