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
#include "sinogram.hpp"
#include "circus.hpp"


//
// Module definitions
//

Transformer::Transformer(const Eigen::MatrixXf &image,
                unsigned int angle_stepsize, bool orthonormal)
                : _image(image), _orthonormal(orthonormal), _angle_stepsize(angle_stepsize)
{
        // Orthonormal P-functionals need a stretched image in order to ensure a
        // square sinogram
        if (_orthonormal) {
                size_t ndiag = (int) std::ceil(360.0/angle_stepsize);
                size_t nsize = (int) std::ceil(ndiag/std::sqrt(2));
                clog(debug) << "Stretching input image to " << nsize << " squared." << std::endl;
                _image = resize(_image, nsize, nsize);
        }

        // Pad the images so we can freely rotate without losing information
        _image = pad(_image);
        clog(debug) << "Padded image to " << _image.rows() << "x" << _image.cols() << std::endl;
}

void Transformer::getTransform(const std::vector<TFunctionalWrapper> &tfunctionals,
                std::vector<PFunctionalWrapper> &pfunctionals, bool write_data) const
{
        // Process all T-functionals
        clog(debug) << "Calculating sinograms for given T-functionals"
                        << std::endl;
        std::vector<Eigen::MatrixXf> sinograms = getSinograms(
                _image,
                _angle_stepsize,
                tfunctionals
        );
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                if (write_data) {
                        // Save the sinogram trace
                        std::stringstream fn_trace_data;
                        fn_trace_data << "trace_" << tfunctionals[t].name << ".csv";
                        writecsv(fn_trace_data.str(), sinograms[t]);

                        if (clog(debug)) {
                                // Save the sinogram image
                                std::stringstream fn_trace_image;
                                fn_trace_image << "trace_" << tfunctionals[t].name << ".pgm";
                                writepgm(fn_trace_image.str(), mat2gray(sinograms[t]));
                        }
                }

                // Orthonormal functionals require the nearest orthonormal sinogram
                if (_orthonormal) {
                        clog(trace) << "Orthonormalizing sinogram" << std::endl;
                        size_t sinogram_center;
                        sinograms[t] = nearest_orthonormal_sinogram(sinograms[t], sinogram_center);
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
                        Eigen::VectorXf circus = getCircusFunction(
                                sinograms[t],
                                pfunctionals[p]
                        );

                        // Normalize
                        Eigen::VectorXf normalized = zscore(circus);

                        if (write_data) {
                                // Save the circus trace
                                std::stringstream fn_trace_data;
                                fn_trace_data << "trace_" << tfunctionals[t].name
                                                << "-" << pfunctionals[p].name << ".csv";
                                writecsv(fn_trace_data.str(), normalized);
                        }
                }
        }
}
