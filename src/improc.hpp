//
// Configuration
//

// Include guard
#ifndef TRACETRANSFORM_IMPROC_HPP
#define TRACETRANSFORM_IMPROC_HPP

// Standard library
#include <cmath>
#include <vector>

// Eigen
#include <Eigen/Dense>

// Local includes
#include "wrapper.hpp"
#include "transform.hpp"
#include "circus.hpp"

// Algorithm parameters
#define ANGLE_INTERVAL          1
#define DISTANCE_INTERVAL       1


//
// Structs
//

struct TFunctional {
        std::string name;
        FunctionalWrapper *wrapper;
};

struct PFunctional
{
        enum
        {
                REGULAR,
                HERMITE
        } type;

        std::string name;
        FunctionalWrapper *wrapper;

        boost::optional<unsigned int> order;
};


//
// Module definitions
//

Eigen::MatrixXd processImage(const Eigen::MatrixXd &input,
                const std::vector<TFunctional> &tfunctionals,
                const std::vector<PFunctional> &pfunctionals,
                bool orthonormal,
                bool verbose)
{
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
                Eigen::MatrixXd sinogram = getTraceTransform(
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

#endif
