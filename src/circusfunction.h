//
// Configuration
//

// Include guard
#ifndef CIRCUSFUNCTION_H
#define CIRCUSFUNCTION_H

// System includes
#include <limits>

// OpenCV includes
#include <cv.h>

// Local includes
#include "auxiliary.h"


//
// Routines
//

cv::Mat getNearestOrthonormalizedSinogram(const cv::Mat &sinogram)
{
	// Detect the offset of each column to the sinogram center
	assert(sinogram.rows > 0);
	unsigned int sinogram_center = (unsigned int) std::floor((sinogram.rows - 1) / 2.0);
	std::vector<int> offset(sinogram.cols);
	for (int p = 0; p < sinogram.cols; p++) {
		// Determine the trace segment
		Segment trace = Segment{
			Point{(double)p, 0},
			Point{(double)p, (double)sinogram.rows-1}
		};

		// Set-up the trace iterator
		TraceIterator<double> iterator(sinogram, trace);
		assert(iterator.valid());

		// Get and compare the median
		Point median = iterator_weighedmedian(iterator);
		offset[p] = (median.y - sinogram_center);
	}

	// Align each column to the sinogram center
	int min = *(std::min_element(offset.begin(), offset.end()));
	int max = *(std::max_element(offset.begin(), offset.end()));
	unsigned int padding = max + std::abs(min);
	cv::Mat aligned = cv::Mat::zeros(
		sinogram.rows + padding,
		sinogram.cols,
		sinogram.type()
	);
	for (int j = 0; j < sinogram.cols; j++) {
		for (int i = 0; i < sinogram.rows; i++) {
			aligned.at<double>(max+i-offset[j], j) = 
				sinogram.at<double>(i, j);
		}
	}

	// Compute the nearest orthonormal sinogram
	cv::SVD svd(aligned, cv::SVD::FULL_UV);
	cv::Mat diagonal = cv::Mat::eye(
		aligned.size(),	// (often) rectangular!
		aligned.type()
	);
	cv::Mat nos = svd.u * diagonal * svd.vt;

	return aligned;
}

cv::Mat getCircusFunction(
	const cv::Mat &sinogram,
	PFunctional<double, double> *functional)
{
	assert(sinogram.type() == CV_64FC1);

	// Create the matrix for the circus functions
	cv::Mat circus(
		cv::Size(sinogram.cols, 1),
		sinogram.type()
	);

	// Apply the P-functional, but check whether we need to process the
	// sinogram directly, or its nearest orthonormalized version
	if (functional->orthonormal()) {
		cv::Mat nos = getNearestOrthonormalizedSinogram(sinogram);
		for (int p = 0; p < nos.cols; p++) {
			// Determine the trace segment
			Segment trace = Segment{
				Point{(double)p, 0},
				Point{(double)p, (double)nos.rows-1}
			};

			// Set-up the trace iterator
			TraceIterator<double> iterator(nos, trace);
			assert(iterator.valid());

			// Apply the functional
			double pixel = (*functional)(iterator);
			circus.at<double>(
				0,	// row
				p	// column
			) = pixel;
		}
	} else {
		for (int p = 0; p < sinogram.cols; p++) {
			// Determine the trace segment
			Segment trace = Segment{
				Point{(double)p, 0},
				Point{(double)p, (double)sinogram.rows-1}
			};

			// Set-up the trace iterator
			TraceIterator<double> iterator(sinogram, trace);
			assert(iterator.valid());

			// Apply the functional
			double pixel = (*functional)(iterator);
			circus.at<double>(
				0,	// row
				p	// column
			) = pixel;
		}
	}

	return circus;
}

#endif
