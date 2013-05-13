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
#include "auxiliary.hpp"
extern "C" {
        #include "functionals.h"
}


//
// Module definitions
//

std::istream& operator>>(std::istream& in, PFunctionalWrapper& wrapper)
{
        in >> wrapper.name;
        if (isdigit(wrapper.name[0]))
            wrapper.name = "P" + wrapper.name;
        if (wrapper.name == "P1") {
                wrapper.functional = PFunctional::P1;
        } else if (wrapper.name == "P2") {
                wrapper.functional = PFunctional::P2;
        } else if (wrapper.name == "P3") {
                wrapper.functional = PFunctional::P3;
        } else if (wrapper.name[0] == 'H') {
                wrapper.functional = PFunctional::Hermite;
                if (wrapper.name.size() < 2)
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Missing order parameter for Hermite P-functional");
                try {
                        wrapper.arguments.order = boost::lexical_cast<unsigned int>(wrapper.name.substr(1));
                }
                catch(boost::bad_lexical_cast &) {
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Unparseable order parameter for Hermite P-functional");
                }
        } else {
                throw boost::program_options::validation_error(
                        boost::program_options::validation_error::invalid_option_value,
                        "Unknown P-functional");
        }
    return in;
}

Eigen::MatrixXf nearest_orthonormal_sinogram(
        const Eigen::MatrixXf &input,
        size_t& new_center)
{
        // Detect the offset of each column to the sinogram center
        assert(input.rows() > 0 && input.cols() > 0);
        int sinogram_center =  std::floor((input.rows() - 1) / 2.0);
        std::vector<int> offset(input.cols());  // TODO: Eigen vector
        for (int p = 0; p < input.cols(); p++) {
                size_t median = findWeighedMedian(
                        input.data() + p*input.rows(),
                        input.rows());
                offset[p] = median - sinogram_center;
        }

        // Align each column to the sinogram center
        int min = *(std::min_element(offset.begin(), offset.end()));
        int max = *(std::max_element(offset.begin(), offset.end()));
        assert(sgn(min) != sgn(max));
        int padding = (int) (std::abs(max) + std::abs(min));
        new_center = sinogram_center + max;
        // TODO: zeros?
        Eigen::MatrixXf aligned(input.rows() + padding, input.cols());
        for (int col = 0; col < input.cols(); col++) {
                for (int row = 0; row < input.rows(); row++) {
                        aligned(max+row-offset[col], col) = input(row, col);
                }
        }

        // Compute the nearest orthonormal sinogram
        Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ColPivHouseholderQRPreconditioner> svd(
                aligned, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXf diagonal = Eigen::MatrixXf::Identity(aligned.rows(), aligned.cols());
        Eigen::MatrixXf nos = svd.matrixU() * diagonal * svd.matrixV().transpose();

        return nos;
}

Eigen::VectorXf getCircusFunction(
        const Eigen::MatrixXf &input,
        const PFunctionalWrapper &pfunctional)
{
        // Allocate the output matrix
        Eigen::VectorXf output(input.cols());

        // Trace all columns
        for (int p = 0; p < input.cols(); p++) {
                float *data = (float*) (input.data() + p*input.rows());
                size_t length = input.rows();
                float result;
                switch (pfunctional.functional) {
                        case PFunctional::P1:
                                result = PFunctional1(data, length);
                                break;
                        case PFunctional::P2:
                                result = PFunctional2(data, length);
                                break;
                        case PFunctional::P3:
                                result = PFunctional3(data, length);
                                break;
                        case PFunctional::Hermite:
                                result = PFunctionalHermite(data, length, *pfunctional.arguments.order, *pfunctional.arguments.center);
                                break;
                }
                output(p) = result;
        }

        return output;
}
