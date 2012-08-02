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

	// Trace all columns
	for (int p = 0; p < sinogram.cols; p++) {
		// Set-up the trace iterator
		ColumnIterator<double> iterator(sinogram, p);
		assert(iterator.valid());

		// Apply the functional
		double pixel = (*functional)(&iterator);
		circus.at<double>(
			0,	// row
			p	// column
		) = pixel;
	}

	return circus;
}

#endif
