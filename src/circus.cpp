//
// Configuration
//

// Header include
#include "circus.hpp"

// Standard library
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>

// Eigen
#include <Eigen/SVD>

// Local
#include "logger.hpp"
extern "C" {
        #include "functionals.h"
}


//
// Module definitions
//

Eigen::MatrixXd nearest_orthonormal_sinogram(
        const Eigen::MatrixXd &input,
        unsigned int& new_center)
{
        // Detect the offset of each column to the sinogram center
        assert(input.rows() > 0 && input.cols() > 0);
        unsigned int sinogram_center = (unsigned int) std::floor((input.rows() - 1) / 2.0);
        std::vector<int> offset(input.cols());  // TODO: Eigen vector
        for (size_t p = 0; p < input.cols(); p++) {
                size_t median = findWeighedMedian(
                        input.data() + p*input.rows(),
                        input.rows());
                offset[p] = median - sinogram_center;
        }

        // Align each column to the sinogram center
        int min = *(std::min_element(offset.begin(), offset.end()));
        int max = *(std::max_element(offset.begin(), offset.end()));
        unsigned int padding = max + std::abs(min);
        new_center = sinogram_center + max;
        // TODO: zeros?
        Eigen::MatrixXd aligned(input.rows() + padding, input.cols());
        for (size_t col = 0; col < input.cols(); col++) {
                for (size_t row = 0; row < input.rows(); row++) {
                        aligned(max+row-offset[col], col) = input(row, col);
                }
        }

        // Compute the nearest orthonormal sinogram
        // NOTE: by not using a QR preconditioner the sinogram HAS to be square
        assert(aligned.rows() == aligned.cols());
        Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::NoQRPreconditioner> svd(
                aligned, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd diagonal = Eigen::MatrixXd::Identity(aligned.rows(), aligned.cols());
        Eigen::MatrixXd nos = svd.matrixU() * diagonal * svd.matrixV().transpose();

        return nos;
}

Eigen::VectorXd getCircusFunction(
        const Eigen::MatrixXd &input,
        FunctionalWrapper *pfunctional)
{
        // Allocate the output matrix
        Eigen::VectorXd output(input.cols());

        // Trace all columns
        #pragma omp parallel for
        for (size_t p = 0; p < input.cols(); p++) {
                output(p) = (*pfunctional)(
                        input.data() + p*input.rows(),
                        input.rows());
        }

        return output;
}
