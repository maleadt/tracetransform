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

Transformer::Transformer(const Eigen::MatrixXd &image)
                : _image(image)
{
        // Orthonormal P-functionals need a stretched image in order to ensure a
        // square sinogram
        size_t ndiag = (int) std::ceil(360.0/ANGLE_INTERVAL);
        size_t nsize = (int) std::ceil(ndiag/std::sqrt(2));
        _image_orthonormal = resize(_image, nsize, nsize);

        // Pad the images so we can freely rotate without losing information
        _image = pad(_image);
        _image_orthonormal = pad(_image_orthonormal);
}

Eigen::MatrixXd Transformer::getTransform(const std::vector<TFunctional> &tfunctionals,
                const std::vector<PFunctional> &pfunctionals) const
{
        // Check for orthonormal P-functionals
        unsigned int orthonormal_count = 0;
        bool orthonormal;
        for (size_t p = 0; p < pfunctionals.size(); p++) {
                if (pfunctionals[p].type == PFunctional::Type::HERMITE)
                        orthonormal_count++;
        }
        if (orthonormal_count == 0)
                orthonormal = false;
        else if (orthonormal_count == pfunctionals.size())
                orthonormal = true;
        else
                throw std::runtime_error(
                        "Cannot mix regular and orthonormal P-functionals");

        // Select an image to use
        const Eigen::MatrixXd *image_selected;
        if (orthonormal)
                image_selected = &_image_orthonormal;
        else
                image_selected = &_image;

        // Allocate a matrix for all output data to reside in
        Eigen::MatrixXd output(
                360 / ANGLE_INTERVAL,
                tfunctionals.size() * pfunctionals.size());

        // Process all T-functionals
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                // Calculate the trace transform sinogram
                clog(debug) << "Calculating " << tfunctionals[t].name << " sinogram" << std::endl;
                Eigen::MatrixXd sinogram = getSinogram(
                        *image_selected,
                        ANGLE_INTERVAL,
                        DISTANCE_INTERVAL,
                        tfunctionals[t].wrapper
                );

                if (clog(debug)) {
                        // Save the sinogram image
                        std::stringstream fn_trace_image;
                        fn_trace_image << "trace_" << tfunctionals[t].name << ".pgm";
                        pgmWrite(fn_trace_image.str(), mat2gray(sinogram));

                        // Save the sinogram data
                        std::stringstream fn_trace_data;
                        fn_trace_data << "trace_" << tfunctionals[t].name << ".dat";
                        dataWrite(fn_trace_data.str(), sinogram);
                }

                // Orthonormal functionals require the nearest orthonormal sinogram
                size_t sinogram_center;
                if (orthonormal) {
                        clog(trace) << "Orthonormalizing sinogram" << std::endl;
                        sinogram = nearest_orthonormal_sinogram(sinogram, sinogram_center);
                }

                // Process all P-functionals
                for (size_t p = 0; p < pfunctionals.size(); p++) {
                        // Configure any extra parameters
                        if (pfunctionals[p].type == PFunctional::Type::HERMITE)
                                dynamic_cast<GenericFunctionalWrapper<unsigned int, size_t>*>
                                        (pfunctionals[p].wrapper)
                                        ->configure(*pfunctionals[p].order, sinogram_center);

                        // Calculate the circus function
                        clog(debug) << "Calculating " << pfunctionals[p].name
                                        << " circus function for "
                                        << tfunctionals[t].name
                                        << " sinogram" << std::endl;
                        Eigen::VectorXd circus = getCircusFunction(
                                sinogram,
                                pfunctionals[p].wrapper
                        );

                        // Normalize
                        Eigen::VectorXd normalized = zscore(circus);

                        // Copy the data
                        assert(normalized.size() == output.rows());
                        output.col(t*pfunctionals.size() + p) = normalized;
                }
        }

        return output;
}
