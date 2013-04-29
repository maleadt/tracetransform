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

CUDAHelper::GlobalMemory<float> *nearest_orthonormal_sinogram(
        const CUDAHelper::GlobalMemory<float>* input,
        size_t& new_center)
{
        // TEMPORARY: download input
        Eigen::MatrixXf input_data(input->size(0), input->size(1));
        input->download(input_data.data());

        // Detect the offset of each column to the sinogram center
        assert(input_data.rows() > 0 && input_data.cols() > 0);
        int sinogram_center =  std::floor((input_data.rows() - 1) / 2.0);
        std::vector<int> offset(input_data.cols());  // TODO: Eigen vector
        for (int p = 0; p < input_data.cols(); p++) {
                size_t median = findWeighedMedian(
                                input_data.data() + p*input_data.rows(),
                                input_data.rows());
                offset[p] = median - sinogram_center;
        }

        // Align each column to the sinogram center
        int min = *(std::min_element(offset.begin(), offset.end()));
        int max = *(std::max_element(offset.begin(), offset.end()));
        assert(sgn(min) != sgn(max));
        int padding = (int) (std::abs(max) + std::abs(min));
        new_center = sinogram_center + max;
        // TODO: zeros?
        Eigen::MatrixXf aligned(input_data.rows() + padding, input_data.cols());
        for (int col = 0; col < input_data.cols(); col++) {
                for (int row = 0; row < input_data.rows(); row++) {
                        aligned(max+row-offset[col], col) = input_data(row, col);
                }
        }

        // Compute the nearest orthonormal sinogram
        Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ColPivHouseholderQRPreconditioner> svd(
                aligned, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXf diagonal = Eigen::MatrixXf::Identity(aligned.rows(), aligned.cols());
        Eigen::MatrixXf nos = svd.matrixU() * diagonal * svd.matrixV().transpose();

        // TEMPORARY: upload input
        CUDAHelper::GlobalMemory<float> *nos_mem = new CUDAHelper::GlobalMemory<float>(CUDAHelper::size_2d(nos.rows(), nos.cols()));
        nos_mem->upload(nos.data());

        return nos_mem;
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
