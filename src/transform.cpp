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


//
// Module definitions
//

#ifdef WITH_CULA
Transformer::Transformer(const Eigen::MatrixXf &image,
                         unsigned int angle_stepsize, bool orthonormal) :
             _orthonormal(orthonormal),
  _angle_stepsize(angle_stepsize) {
#else
Transformer::Transformer(const Eigen::MatrixXf &image,
                         unsigned int angle_stepsize) :
#endif
    _image(image), _angle_stepsize(angle_stepsize) {
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
    // Process all T-functionals
    clog(debug) << "Calculating sinograms for given T-functionals" << std::endl;
    std::vector<CUDAHelper::GlobalMemory<float> *> sinograms =
        getSinograms(_memory, _angle_stepsize, tfunctionals);
    for (size_t t = 0; t < tfunctionals.size(); t++) {
        if (write_data) {
            // Download the image
            Eigen::MatrixXf sinogram_data(sinograms[t]->size(0),
                                          sinograms[t]->size(1));
            sinograms[t]->download(sinogram_data.data());

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
            clog(debug)
                << "Calculating circus functions for given P-functionals"
                << std::endl;
            std::vector<CUDAHelper::GlobalMemory<float> *> circusfunctions =
                getCircusFunctions(sinograms[t], pfunctionals);
        for (size_t p = 0; p < pfunctionals.size(); p++) {
            // TEMPORARY: download the image
            Eigen::VectorXf circusfunction_data(circusfunctions[p]->size(0));
            circusfunctions[p]->download(circusfunction_data.data());

            // Normalize
            Eigen::VectorXf normalized = zscore(circusfunction_data);

            if (write_data) {
                // Save the circus trace
                std::stringstream fn_trace_data;
                fn_trace_data << "trace_" << tfunctionals[t].name << "-"
                              << pfunctionals[p].name << ".csv";
                writecsv(fn_trace_data.str(), normalized);
            }
        }

        // Clean-up
        for (size_t p = 0; p < pfunctionals.size(); p++)
            delete circusfunctions[p];
    }

    // Clean-up
    for (size_t t = 0; t < tfunctionals.size(); t++)
        delete sinograms[t];
}
