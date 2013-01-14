//
// Configuration
//

// Include guard
#ifndef CIRCUSFUNCTION_H
#define CIRCUSFUNCTION_H

// Standard library
#include <limits>

// Eigen
#include <Eigen/Dense>
#include <Eigen/SVD>

// Local includes
#include "auxiliary.h"
#include "wrapper.h"
extern "C" {
	#include "functionals.h"
}


//
// Routines
//

Eigen::MatrixXd nearest_orthonormal_sinogram(
	const Eigen::MatrixXd &input,
	unsigned int& new_center)
{
	// Detect the offset of each column to the sinogram center
	assert(input.rows() > 0);
	assert(input.cols() >= 0);
	unsigned int sinogram_center = (unsigned int) std::floor((input.rows() - 1) / 2.0);
	std::vector<int> offset(input.cols());	// TODO: Eigen vector
	for (int p = 0; p < input.cols(); p++) {
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
	Eigen::MatrixXd aligned(input.rows() + padding, input.cols());
	for (int col = 0; col < input.cols(); col++) {
		for (int row = 0; row < input.rows(); row++) {
			aligned(max+row-offset[col], col) = input(row, col);
		}
	}

	// Compute the nearest orthonormal sinogram
	Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner> svd(
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
	for (int p = 0; p < input.cols(); p++) {
		output(p) = (*pfunctional)(
			input.data() + p*input.rows(),
			input.rows());
	}

	return output;
}

#endif
