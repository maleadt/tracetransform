//
// Configuration
//

// Header include
#include "transform.hpp"

// Standard library
#include <cmath>
#include <vector>
#include <chrono>
#include <ostream>
#include <stdexcept>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "sinogram.hpp"
#include "circus.hpp"


//
// Module definitions
//

Transformer::Transformer(const Eigen::MatrixXf &image)
                : _image(image)
{
        // Orthonormal P-functionals need a stretched image in order to ensure a
        // square sinogram
        size_t ndiag = 360;
        size_t nsize = (int) std::ceil(ndiag/std::sqrt(2));
        _image_orthonormal = resize(_image, nsize, nsize);

        // Pad the images so we can freely rotate without losing information
        _image = pad(_image);
        _image_orthonormal = pad(_image_orthonormal);
}

Eigen::MatrixXf Transformer::getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                std::vector<PFunctionalWrapper> &pfunctionals) const
{
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

        // Check for orthonormal P-functionals
        unsigned int orthonormal_count = 0;
        bool orthonormal;
        for (size_t p = 0; p < pfunctionals.size(); p++) {
                if (pfunctionals[p].functional == PFunctional::Hermite)
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
        const Eigen::MatrixXf *image_selected;
        if (orthonormal)
                image_selected = &_image_orthonormal;
        else
                image_selected = &_image;

        // Allocate a matrix for all output data to reside in
        Eigen::MatrixXf output(
                360,
                tfunctionals.size() * pfunctionals.size());

        // Process all T-functionals
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                // Calculate the trace transform sinogram
                clog(debug) << "Calculating " << tfunctionals[t].name << " sinogram" << std::endl;
                start = std::chrono::system_clock::now();
                Eigen::MatrixXf sinogram = getSinogram(
                        *image_selected,
                        tfunctionals[t]
                );
                end = std::chrono::system_clock::now();
                clog(debug) << "Sinogram calculation took "
                                << std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()
                                << " ms." << std::endl;

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
                if (orthonormal) {
                        clog(trace) << "Orthonormalizing sinogram" << std::endl;
                        size_t sinogram_center;
                        sinogram = nearest_orthonormal_sinogram(sinogram, sinogram_center);
                        for (size_t p = 0; p < pfunctionals.size(); p++) {
                                if (pfunctionals[p].functional == PFunctional::Hermite) {
                                        pfunctionals[p].arguments.center = sinogram_center;
                                }
                        }
                }

                // Process all P-functionals
                for (size_t p = 0; p < pfunctionals.size(); p++) {
                        // Calculate the circus function
                        clog(debug) << "Calculating " << pfunctionals[p].name
                                        << " circus function for "
                                        << tfunctionals[t].name
                                        << " sinogram" << std::endl;
                        Eigen::VectorXf circus = getCircusFunction(
                                sinogram,
                                pfunctionals[p]
                        );

                        // Normalize
                        Eigen::VectorXf normalized = zscore(circus);

                        // Copy the data
                        assert(normalized.size() == output.rows());
                        output.col(t*pfunctionals.size() + p) = normalized;
                }
        }

        return output;
}
