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
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
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
        Transformer(const Eigen::MatrixXd &image,
                const std::vector<TFunctional> &tfunctionals,
                const std::vector<PFunctional> &pfunctionals);

        Eigen::MatrixXd getTransform() const;

private:
        const std::vector<TFunctional> &_tfunctionals;
        const std::vector<PFunctional> &_pfunctionals;
        Eigen::MatrixXd _image;
        bool _orthonormal;
};

#endif
