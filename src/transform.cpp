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
#ifdef WITH_CULA
#include "kernels/nos.hpp"
#endif
#include "kernels/stats.hpp"


//
// Module definitions
//

#ifdef WITH_CULA
Transformer::Transformer(const Eigen::MatrixXf &image,
                         const std::string &basename,
                         unsigned int angle_stepsize, bool orthonormal)
    : _orthonormal(orthonormal), _angle_stepsize(angle_stepsize) {
#else
Transformer::Transformer(const Eigen::MatrixXf &image, const std::string &basename,
                         unsigned int angle_stepsize) :
#endif
    _image(image), _basename(basename), _angle_stepsize(angle_stepsize) {
#ifdef WITH_CULA
    // Orthonormal P-functionals need a stretched image in order to ensure a
    // square sinogram
    if (_orthonormal) {
        size_t ndiag = (int)std::ceil(360.0 / angle_stepsize);
        size_t nsize = (int)std::ceil(ndiag / std::sqrt(2));
        clog(debug) << "Stretching input image to " << nsize << " squared."
                    << std::endl;
        _image = resize(_image, nsize, nsize);
    }
#endif

    // Pad the images so we can freely rotate without losing information
    _image = pad(_image);
    clog(debug) << "Padded image to " << _image.rows() << "x" << _image.cols()
                << std::endl;

    // Upload the image to device memory
    // TODO: rename _memory
    _memory = new CUDAHelper::GlobalMemory<float>(
        CUDAHelper::size_2d(_image.rows(), _image.cols()));
    _memory->upload(_image.data());
}

void
Transformer::getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                          std::vector<PFunctionalWrapper> &pfunctionals,
                          bool write_data) const {
    Eigen::MatrixXf signatures((int)std::floor(360 / _angle_stepsize),
                               tfunctionals.size() * pfunctionals.size());

    // Process all T-functionals
    clog(debug) << "Calculating sinograms for given T-functionals" << std::endl;
    std::vector<CUDAHelper::GlobalMemory<float> *> sinograms =
        getSinograms(_memory, _angle_stepsize, tfunctionals);
    for (size_t t = 0; t < tfunctionals.size(); t++) {
        if (write_data && clog(debug)) {
            // Download the image
            Eigen::MatrixXf sinogram_data(sinograms[t]->size(0),
                                          sinograms[t]->size(1));
            sinograms[t]->download(sinogram_data.data());

            // Save the sinogram trace
            std::stringstream fn_trace_data;
            fn_trace_data << _basename << "-" << tfunctionals[t].name << ".csv";
            writecsv(fn_trace_data.str(), sinogram_data);

            // Save the sinogram image
            std::stringstream fn_trace_image;
            fn_trace_image << _basename << "-" << tfunctionals[t].name
                           << ".pgm";
            writepgm(fn_trace_image.str(), mat2gray(sinogram_data));
        }

#ifdef WITH_CULA
        // Orthonormal functionals require the nearest orthonormal sinogram
        if (_orthonormal) {
            clog(trace) << "Orthonormalizing sinogram" << std::endl;
            size_t sinogram_center;
            CUDAHelper::GlobalMemory<float> *nos =
                nearest_orthonormal_sinogram(sinograms[t], sinogram_center);
            delete sinograms[t];
            sinograms[t] = nos;
            for (size_t p = 0; p < pfunctionals.size(); p++) {
                if (pfunctionals[p].functional == PFunctional::Hermite) {
                    pfunctionals[p].arguments.center = sinogram_center;
                }
            }
        }
#endif

        // Process all P-functionals
        if (pfunctionals.size() > 0) {
            clog(debug)
                << "Calculating circus functions for given P-functionals"
                << std::endl;
            std::vector<CUDAHelper::GlobalMemory<float> *> circusfunctions =
                getCircusFunctions(sinograms[t], pfunctionals);

            for (size_t p = 0; p < pfunctionals.size(); p++) {
                // Normalize
                CUDAHelper::GlobalMemory<float> *normalized =
                    zscore(circusfunctions[p]);

                if (write_data) {
                    // Download the trace
                    Eigen::VectorXf normalized_data(circusfunctions[p]->size(0));
                    normalized->download(normalized_data.data());

                    // Aggregate the signatures
                    assert(signatures.rows() == normalized_data.size());
                    signatures.col(t * pfunctionals.size() + p) = normalized_data;
                }

                delete normalized;
            }

            // Clean-up
            for (size_t p = 0; p < pfunctionals.size(); p++)
                delete circusfunctions[p];
        }
    }

    // Clean-up
    cudaDeviceSynchronize();
    for (size_t t = 0; t < tfunctionals.size(); t++)
        delete sinograms[t];

    // Save the signatures
    if (write_data && pfunctionals.size() > 0) {
        std::stringstream fn_signatures;
        fn_signatures << _basename << ".csv";
        writecsv(fn_signatures.str(), signatures);
    }
}
