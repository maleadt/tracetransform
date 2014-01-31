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
#include "cudahelper/memory.hpp"
#include "sinogram.hpp"
#include "circus.hpp"


//
// Module definitions
//

class Transformer {
  public:
#ifdef WITH_CULA
    Transformer(const Eigen::MatrixXf &image, const std::string &basename,
                unsigned int angle_step, bool orthonormal);
#else
    Transformer(const Eigen::MatrixXf &image, const std::string &basename,
                unsigned int angle_step);
#endif

    void getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                      std::vector<PFunctionalWrapper> &pfunctionals,
                      bool write_data = true) const;

  private:
    Eigen::MatrixXf _image;
    std::string _basename;
    unsigned int _angle_stepsize;
    CUDAHelper::GlobalMemory<float> *_memory;
#ifdef WITH_CULA
    bool _orthonormal;
#endif
};

#endif
