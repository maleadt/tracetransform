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

        TFunctional(std::string name_, FunctionalWrapper *wrapper_)
                : name(name_), wrapper(wrapper_)
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

        PFunctional(std::string name_, FunctionalWrapper *wrapper_,
                        Type type_ = Type::REGULAR, boost::optional<unsigned int> order_ = boost::none)
                : name(name_), wrapper(wrapper_), type(type_), order(order_)
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
        Transformer(const Eigen::MatrixXf &image);

        Eigen::MatrixXf getTransform(const std::vector<TFunctional> &tfunctionals,
                        const std::vector<PFunctional> &pfunctionals) const;

private:
        Eigen::MatrixXf _image, _image_orthonormal;
};

#endif
