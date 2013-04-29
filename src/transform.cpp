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


//
// Module definitions
//

Transformer::Transformer(const Eigen::MatrixXf &image, bool orthonormal)
                : _image(image), _orthonormal(orthonormal)
{
        // Orthonormal P-functionals need a stretched image in order to ensure a
        // square sinogram
        if (_orthonormal) {
                size_t ndiag = 360;
                size_t nsize = (int) std::ceil(ndiag/std::sqrt(2));
                _image = resize(_image, nsize, nsize);
        }

        // Pad the images so we can freely rotate without losing information
        _image = pad(_image);

        // Upload the image to device memory
        _memory = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(
                        _image.rows(), _image.cols()));
        _memory->upload(_image.data());

}

Eigen::MatrixXf Transformer::getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                std::vector<PFunctionalWrapper> &pfunctionals) const
{
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

        // Allocate a matrix for all output data to reside in
        Eigen::MatrixXf output(
                360,
                tfunctionals.size() * pfunctionals.size());

        // Process all T-functionals
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                // Calculate the trace transform sinogram
                clog(debug) << "Calculating " << tfunctionals[t].name << " sinogram" << std::endl;
                start = std::chrono::system_clock::now();
                CUDAHelper::GlobalMemory<float> *sinogram = getSinogram(
                        _memory, tfunctionals[t]
                );
                end = std::chrono::system_clock::now();
                clog(debug) << "Sinogram calculation took "
                                << std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count()
                                << " ms." << std::endl;

                // TEMPORARY: download image
                Eigen::MatrixXf sinogram_data(sinogram->size(0), sinogram->size(1));
                sinogram->download(sinogram_data.data());
                delete sinogram;

                if (clog(debug)) {
                        // Save the sinogram image
                        std::stringstream fn_trace_image;
                        fn_trace_image << "trace_" << tfunctionals[t].name << ".pgm";
                        pgmWrite(fn_trace_image.str(), mat2gray(sinogram_data));

                        // Save the sinogram data
                        std::stringstream fn_trace_data;
                        fn_trace_data << "trace_" << tfunctionals[t].name << ".dat";
                        dataWrite(fn_trace_data.str(), sinogram_data);
                }

                // Orthonormal functionals require the nearest orthonormal sinogram
                if (_orthonormal) {
                        clog(trace) << "Orthonormalizing sinogram" << std::endl;
                        size_t sinogram_center;
                        sinogram_data = nearest_orthonormal_sinogram(sinogram_data, sinogram_center);
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
                                sinogram_data,
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
