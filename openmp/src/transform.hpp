//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_TRANSFORM_
#define _TRACETRANSFORM_TRANSFORM_

// Standard library
#include <vector> // for vector

// Eigen
#include <Eigen/Dense>

// Local
#include "sinogram.hpp"
#include "circus.hpp"


//
// Module definitions
//

class Transformer {
  public:
    Transformer(const Eigen::MatrixXf &image, const std::string &basename,
                unsigned int angle_step, bool orthonormal);

    void getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                      std::vector<PFunctionalWrapper> &pfunctionals,
                      bool write_data = true) const;

  private:
    Eigen::MatrixXf _image;
    std::string _basename;
    bool _orthonormal;
    unsigned int _angle_stepsize;
};

#endif
