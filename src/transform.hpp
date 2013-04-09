//
// Configuration
//

// Include guard
#ifndef TRACETRANSFORM_TRANSFORM_HPP
#define TRACETRANSFORM_TRANSFORM_HPP

// Standard library
#include <vector>

// Boost
#include <boost/optional.hpp>

// Eigen
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
#include <Eigen/Dense>

// Local
#include "wrapper.hpp"


//
// Structs
//

struct TFunctional {
        std::string name;
        FunctionalWrapper *wrapper;
};

struct PFunctional
{
        enum
        {
                REGULAR,
                HERMITE
        } type;

        std::string name;
        FunctionalWrapper *wrapper;

        boost::optional<unsigned int> order;
};


//
// Module definitions
//

Eigen::MatrixXd getTransform(/*const*/ Eigen::MatrixXd &input,
                const std::vector<TFunctional> &tfunctionals,
                const std::vector<PFunctional> &pfunctionals,
                bool verbose);

#endif
