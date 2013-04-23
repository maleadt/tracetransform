//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_TRANSFORM_
#define _TRACETRANSFORM_TRANSFORM_

// Standard library
#include <vector>

// Boost
#include <boost/optional.hpp>

// Eigen
#include <Eigen/Dense>

// Local
#include "sinogram.hpp"
#include "circus.hpp"


//
// Module definitions
//

class Transformer
{
public:
        Transformer(const Eigen::MatrixXf &image);

        Eigen::MatrixXf getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                        std::vector<PFunctionalWrapper> &pfunctionals) const;

private:
        Eigen::MatrixXf _image, _image_orthonormal;
};

#endif
