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

cv::Mat getOrthonormalCircusFunction(
	const cv::Mat &sinogram,
	Functional functional)
{
	assert(sinogram.type() == CV_64FC1);

	// Change the domain of the sinogram from p to z
	// TODO
	cv::Mat transformed = sinogram;

	// Obtain the nearest orthonormal sinogram
	cv::SVD svd(transformed);
	cv::Mat Sk = cv::Mat::eye(
		transformed.size(),
		transformed.type()
	);
	cv::Mat nos = svd.u * Sk * svd.vt;

	// Apply the Laguerre P-functional
	cv::Mat circus(
		cv::Size(transformed.cols, 1),
		transformed.type()
	);
	for (int p = 0; p < transformed.cols; p++) {
		// Determine the trace segment
		Segment trace = Segment{
			Point{(double)p, 0},
			Point{(double)p, (double)transformed.rows-1}
		};

		// Set-up the trace iterator
		TraceIterator iterator(transformed, trace);
		assert(iterator.valid());

		// Apply the functional
		double pixel = functional(iterator);
		circus.at<double>(
			p,	// row
			0	// column
		) = pixel;
	}

	return circus;
}

#endif
