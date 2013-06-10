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
#include "kernels/nos.hpp"


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
                clog(debug) << "Stretching input image to " << nsize << " squared." << std::endl;
                _image = resize(_image, nsize, nsize);
        }

        // Pad the images so we can freely rotate without losing information
        _image = pad(_image);
        clog(debug) << "Padded image to " << _image.rows() << "x" << _image.cols() << std::endl;

        // Upload the image to device memory
        _memory = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(
                        _image.rows(), _image.cols()));
        _memory->upload(_image.data());
}

void Transformer::getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                std::vector<PFunctionalWrapper> &pfunctionals, bool write_data) const
{
        // Process all T-functionals
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                // Calculate the trace transform sinogram
                clog(debug) << "Calculating sinogram using T-functional "
                                << tfunctionals[t].name << std::endl;
                CUDAHelper::GlobalMemory<float> *sinogram = getSinogram(
                        _memory,
                        tfunctionals[t]
                );

                if (write_data) {
                        // Download the image
                        Eigen::MatrixXf sinogram_data(sinogram->size(0), sinogram->size(1));
                        sinogram->download(sinogram_data.data());

                        // Save the sinogram trace
                        std::stringstream fn_trace_data;
                        fn_trace_data << "trace_" << tfunctionals[t].name << ".csv";
                        writecsv(fn_trace_data.str(), sinogram_data);

                        if (clog(debug)) {
                                // Save the sinogram image
                                std::stringstream fn_trace_image;
                                fn_trace_image << "trace_" << tfunctionals[t].name << ".pgm";
                                writepgm(fn_trace_image.str(), mat2gray(sinogram_data));
                        }
                }

                // Orthonormal functionals require the nearest orthonormal sinogram
                if (_orthonormal) {
                        clog(trace) << "Orthonormalizing sinogram" << std::endl;
                        size_t sinogram_center;
                        CUDAHelper::GlobalMemory<float> *nos = nearest_orthonormal_sinogram(sinogram, sinogram_center);
                        delete sinogram;
                        sinogram = nos;
                        for (size_t p = 0; p < pfunctionals.size(); p++) {
                                if (pfunctionals[p].functional == PFunctional::Hermite) {
                                        pfunctionals[p].arguments.center = sinogram_center;
                                }
                        }
                }

                // Process all P-functionals
                for (size_t p = 0; p < pfunctionals.size(); p++) {
                        // Calculate the circus function
                        clog(debug) << "Calculating circus function using P-functional "
                                        << pfunctionals[p].name
                                        << std::endl;
                        CUDAHelper::GlobalMemory<float> *circus = getCircusFunction(
                                sinogram,
                                pfunctionals[p]
                        );

                        // TEMPORARY: download the image
                        Eigen::VectorXf circus_data(circus->size(0));
                        circus->download(circus_data.data());
                        delete circus;

                        // Normalize
                        Eigen::VectorXf normalized = zscore(circus_data);

                        if (write_data) {
                                // Save the circus trace
                                std::stringstream fn_trace_data;
                                fn_trace_data << "trace_" << tfunctionals[t].name
                                                << "-" << pfunctionals[p].name << ".csv";
                                writecsv(fn_trace_data.str(), normalized);
                        }
                }

                // Clean-up
                delete sinogram;
        }
}
