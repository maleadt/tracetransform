//
// Configuration
//

// Header include
#include "transform.hpp"

// Standard library
#include <cmath>
#include <vector>
#include <ostream>
#include <stdexcept>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "wrapper.hpp"
#include "sinogram.hpp"
#include "circus.hpp"

// Algorithm parameters
#define ANGLE_INTERVAL          1
#define DISTANCE_INTERVAL       1


//
// Module definitions
//

Transformer::Transformer(const Eigen::MatrixXd &image,
                const std::vector<TFunctional> &tfunctionals,
                const std::vector<PFunctional> &pfunctionals)
                : _tfunctionals(tfunctionals), _pfunctionals(pfunctionals)
{
        // Check for orthonormal P-functionals
        unsigned int orthonormal_count = 0;
        for (size_t p = 0; p < _pfunctionals.size(); p++) {
                if (_pfunctionals[p].type == PFunctional::HERMITE)
                        orthonormal_count++;
        }
        if (orthonormal_count == 0)
                _orthonormal = false;
        else if (orthonormal_count == _pfunctionals.size())
                _orthonormal = true;
        else
                throw std::runtime_error(
                        "Cannot mix regular and orthonormal P-functionals");

        // Orthonormal P-functionals need a stretched image in order to ensure a
        // square sinogram
        if (_orthonormal) {
                int ndiag = (int) std::ceil(360.0/ANGLE_INTERVAL);
                int nsize = (int) std::ceil(ndiag/std::sqrt(2));
                _image = resize(image, nsize, nsize);
        } else {
                _image = image;
        }

        // Pad the image so we can freely rotate without losing information
        Point origin(
                std::floor((_image.cols() + 1) / 2.0) - 1,
                std::floor((_image.rows() + 1) / 2.0) - 1);
        int rLast = (int) std::ceil(std::hypot(
                _image.cols() - 1 - origin.x() - 1,
                _image.rows() - 1 - origin.y() - 1)) + 1;
        int rFirst = -rLast;
        int nBins = rLast - rFirst + 1;
        Eigen::MatrixXd image_padded = Eigen::MatrixXd::Zero(nBins, nBins);
        Point origin_padded(
                std::floor((image_padded.cols() + 1) / 2.0) - 1,
                std::floor((image_padded.rows() + 1) / 2.0) - 1);
        Point df = origin_padded - origin;
        for (size_t col = 0; col < _image.cols(); col++) {
                for (size_t row = 0; row < _image.rows(); row++) {
                        image_padded(row + (int) df.y(), col + (int) df.x())
                                = _image(row, col);
                }
        }
        _image = image_padded;
}

Eigen::MatrixXd Transformer::getTransform() const
{
        // Allocate a matrix for all output data to reside in
        Eigen::MatrixXd output(
                360 / ANGLE_INTERVAL,
                _tfunctionals.size() * _pfunctionals.size());

        // Process all T-functionals
        for (size_t t = 0; t < _tfunctionals.size(); t++) {
                // Calculate the trace transform sinogram
                clog(debug) << "Calculating sinogram " << _tfunctionals[t].name << std::endl;
                Eigen::MatrixXd sinogram = getSinogram(
                        _image,
                        ANGLE_INTERVAL,
                        DISTANCE_INTERVAL,
                        _tfunctionals[t].wrapper
                );

                if (clog(debug)) {
                        // Save the sinogram image
                        std::stringstream fn_trace_image;
                        fn_trace_image << "trace_" << _tfunctionals[t].name << ".pgm";
                        pgmWrite(fn_trace_image.str(), mat2gray(sinogram));

                        // Save the sinogram data
                        std::stringstream fn_trace_data;
                        fn_trace_data << "trace_" << _tfunctionals[t].name << ".dat";
                        dataWrite(fn_trace_data.str(), sinogram);
                }

                // Orthonormal functionals require the nearest orthonormal sinogram
                unsigned int sinogram_center;
                if (_orthonormal)
                        sinogram = nearest_orthonormal_sinogram(sinogram, sinogram_center);

                // Process all P-functionals
                for (size_t p = 0; p < _pfunctionals.size(); p++) {
                        // Configure any extra parameters
                        if (_pfunctionals[p].type == PFunctional::HERMITE)
                                dynamic_cast<GenericFunctionalWrapper<unsigned int, unsigned int>*>
                                        (_pfunctionals[p].wrapper)
                                        ->configure(*_pfunctionals[p].order, sinogram_center);

                        // Calculate the circus function
                        clog(debug) << "Calculating circus function" << _tfunctionals[t].name << "-" << _pfunctionals[p].name << std::endl;
                        Eigen::VectorXd circus = getCircusFunction(
                                sinogram,
                                _pfunctionals[p].wrapper
                        );

                        // Normalize
                        Eigen::VectorXd normalized = zscore(circus);

                        // Copy the data
                        assert(normalized.size() == output.rows());
                        output.col(t*_pfunctionals.size() + p) = normalized;
                }
        }

        return output;
}
