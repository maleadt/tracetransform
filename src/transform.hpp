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
#include "wrapper.hpp"


//
// Structs
//

struct TFunctional {
        TFunctional()
                : wrapper(nullptr)
        {
        }

        TFunctional(std::string name, FunctionalWrapper *wrapper)
                : name(name), wrapper(wrapper)
        {
        }

        std::string name;
        FunctionalWrapper *wrapper;
};

struct PFunctional
{
        enum class Type
        {
                REGULAR,
                HERMITE
        };

        PFunctional()
                : wrapper(nullptr)
        {
        }

        PFunctional(std::string name, FunctionalWrapper *wrapper,
                        Type type = Type::REGULAR, boost::optional<unsigned int> order = boost::none)
                : name(name), wrapper(wrapper), type(type), order(order)
        {
        }

        std::string name;
        FunctionalWrapper *wrapper;
        Type type;
        boost::optional<unsigned int> order;
};


//
// Module definitions
//

class Transformer
{
public:
        Transformer(const Eigen::MatrixXd &image);

        Eigen::MatrixXd getTransform(const std::vector<TFunctional> &tfunctionals,
                        const std::vector<PFunctional> &pfunctionals) const;

private:
        Eigen::MatrixXd _image, _image_orthonormal;
};

#endif
