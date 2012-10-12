//
// Configuration
//

// Include guard
#ifndef CIRCUSFUNCTION_H
#define CIRCUSFUNCTION_H

// System includes
#include <limits>

// Library includes
#include <cv.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

// Local includes
#include "auxiliary.h"


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
	for (unsigned int p = 0; p < input.cols(); p++) {
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
		for (unsigned int row = 0; row < input.rows(); row++) {
			aligned(max+row-offset[col], col) = input(row, col);
		}
	}

	// Compute the nearest orthonormal sinogram
	Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner> svd(
		aligned, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXd diagonal = Eigen::MatrixXd::Identity(aligned.rows(), aligned.cols());
	Eigen::MatrixXd nos = svd.matrixU() * diagonal * svd.matrixV();

	return nos;
}

cv::Mat getCircusFunction(
	const cv::Mat &input,
	Functional pfunctional,
	void* pfunctional_arguments)
{
	assert(input.type() == CV_64FC1);

	// Transpose the input since cv::Mat is stored in row-major order
	cv::Mat input_transposed;
	cv::transpose(input, input_transposed);

	// Allocate the output matrix
	cv::Mat output(
		cv::Size(input.cols, 1),
		input.type()
	);

	// Trace all columns
	for (int p = 0; p < input.cols; p++) {
		output.at<double>(
			0,	// row
			p	// column
		) = pfunctional(
			input_transposed.ptr<double>(p),
			input_transposed.cols,
			pfunctional_arguments);
	}

	return output;
}

#endif
