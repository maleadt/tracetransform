//
// Configuration
//

// Include guard
#ifndef ORTHOCIRCUSFUNCTION_H
#define ORTHOCIRCUSFUNCTION_H

// System includes
#include <limits>

// OpenCV includes
#include <cv.h>

// Local includes
#include "auxiliary.h"


//
// Routines
//

cv::Mat getCircusFunction(
	const cv::Mat &sinogram,
	Functional<double, double> functional)
{
	assert(sinogram.type() == CV_64FC1);

	// Apply the P-functional
	cv::Mat circus(
		cv::Size(sinogram.cols, 1),
		sinogram.type()
	);
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
		double pixel = functional(iterator);
		circus.at<double>(
			0,	// row
			p	// column
		) = pixel;
	}

	return circus;
}

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

// Hn(g(p)) = Int psi(z)
// TODO: this shouldn't be a special-case functional, but should fit the generic
//       calling semantics. Again, class inheritance to pass arbitrary data?
template <typename T>
double pfunctional_hermite(TraceIterator<T> &iterator, unsigned int degree)
{
	// Discretize the [-10, 10] domain to fit the column iterator
	double z = -10;
	double stepsize = 20.0 / (iterator.samples() - 1);

	// Calculate the integral
	double integral = 0;
	while (iterator.hasNext()) {
		integral += iterator.value() * hermite_function(degree, z);
		iterator.next();
		z += stepsize;
	}
	return integral;
}

cv::Mat getHermiteCircusFunction(
	const cv::Mat &sinogram,
	unsigned int degree)
{
	assert(sinogram.type() == CV_64FC1);

	// Get the nearest orthonormal sinogram
	cv::Mat nos = getNearestOrthonormalizedSinogram(sinogram);

	// Apply the P-functional
	cv::Mat circus(
		cv::Size(sinogram.cols, 1),
		sinogram.type()
	);
	for (int p = 0; p < sinogram.cols; p++) {
		// Determine the trace segment
		Segment trace = Segment{
			Point{(double)p, 0},
			Point{(double)p, (double)sinogram.rows-1}
		};

		// Set-up the trace iterator
		TraceIterator<double> iterator(nos, trace);
		assert(iterator.valid());

		// Apply the functional
		double pixel = pfunctional_hermite(iterator, degree);
		circus.at<double>(
			0,	// row
			p	// column
		) = pixel;
	}

	return circus;
}

#endif
