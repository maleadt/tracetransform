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
#include "wrapper.hpp"
#include "sinogram.hpp"
#include "circus.hpp"
#include "auxiliary.hpp"

// Algorithm parameters
#define ANGLE_INTERVAL          1
#define DISTANCE_INTERVAL       1


//
// Module definitions
//

Eigen::MatrixXd getTransform(/*const*/ Eigen::MatrixXd &input,
                const std::vector<TFunctional> &tfunctionals,
                const std::vector<PFunctional> &pfunctionals,
                bool verbose)
{
        // Check for orthonormal P-functionals
        unsigned int orthonormal_count = 0;
        for (size_t p = 0; p < pfunctionals.size(); p++) {
                if (pfunctionals[p].type == PFunctional::HERMITE)
                        orthonormal_count++;
        }
        bool orthonormal;
        if (orthonormal_count == 0)
                orthonormal = false;
        else if (orthonormal_count == pfunctionals.size())
                orthonormal = true;
        else
                throw std::runtime_error(
                        "Cannot mix regular and orthonormal P-functionals");

        // Orthonormal P-functionals need a stretched image in order to ensure a
        // square sinogram
        if (orthonormal) {
                int ndiag = (int) std::ceil(360.0/ANGLE_INTERVAL);
                int nsize = (int) std::ceil(ndiag/std::sqrt(2));
                input = resize(input, nsize, nsize);
        }

        // Pad the image so we can freely rotate without losing information
        Point origin(
                std::floor((input.cols() + 1) / 2.0) - 1,
                std::floor((input.rows() + 1) / 2.0) - 1);
        int rLast = (int) std::ceil(std::hypot(
                input.cols() - 1 - origin.x() - 1,
                input.rows() - 1 - origin.y() - 1)) + 1;
        int rFirst = -rLast;
        int nBins = rLast - rFirst + 1;
        Eigen::MatrixXd input_padded = Eigen::MatrixXd::Zero(nBins, nBins);
        Point origin_padded(
                std::floor((input_padded.cols() + 1) / 2.0) - 1,
                std::floor((input_padded.rows() + 1) / 2.0) - 1);
        Point df = origin_padded - origin;
        for (size_t col = 0; col < input.cols(); col++) {
                for (size_t row = 0; row < input.rows(); row++) {
                        input_padded(row + (int) df.y(), col + (int) df.x())
                                = input(row, col);
                }
        }

        // Allocate a matrix for all output data to reside in
        Eigen::MatrixXd output(
                360 / ANGLE_INTERVAL,
                tfunctionals.size() * pfunctionals.size());

        // Process all T-functionals
        if (verbose)
                std::cerr << "Calculating";
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                // Calculate the trace transform sinogram
                if (verbose)
                        std::cerr << " " << tfunctionals[t].name << "..." << std::flush;
                Eigen::MatrixXd sinogram = getSinogram(
                        input_padded,
                        ANGLE_INTERVAL,
                        DISTANCE_INTERVAL,
                        tfunctionals[t].wrapper
                );

#ifndef NDEBUG
                // Save the sinogram image
                std::stringstream fn_trace_image;
                fn_trace_image << "trace_" << tfunctionals[t].name << ".pgm";
                pgmWrite(fn_trace_image.str(), mat2gray(sinogram));

                // Save the sinogram data
                std::stringstream fn_trace_data;
                fn_trace_data << "trace_" << tfunctionals[t].name << ".dat";
                dataWrite(fn_trace_data.str(), sinogram);
#endif

                // Orthonormal functionals require the nearest orthonormal sinogram
                unsigned int sinogram_center;
                if (orthonormal)
                        sinogram = nearest_orthonormal_sinogram(sinogram, sinogram_center);

                // Process all P-functionals
                for (size_t p = 0; p < pfunctionals.size(); p++) {
                        // Configure any extra parameters
                        if (pfunctionals[p].type == PFunctional::HERMITE)
                                dynamic_cast<GenericFunctionalWrapper<unsigned int, unsigned int>*>
                                        (pfunctionals[p].wrapper)
                                        ->configure(*pfunctionals[p].order, sinogram_center);

                        // Calculate the circus function
                        if (verbose)
                                std::cerr << " " << pfunctionals[p].name << "..." << std::flush;
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
        if (verbose)
                std::cerr << "\n";

        return output;
}
