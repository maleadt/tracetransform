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

	// Preprocess the sinogram
	bool preprocessed = false;
	const cv::Mat *actual_sinogram = functional->preprocess(sinogram);
	if (actual_sinogram == nullptr)
		actual_sinogram = &sinogram;
	else
		preprocessed = true;

	// Create the matrix for the circus functions
	cv::Mat circus(
		cv::Size(actual_sinogram->cols, 1),
		actual_sinogram->type()
	);

	// Trace all columns
	for (int p = 0; p < actual_sinogram->cols; p++) {
		// Determine the trace segment
		Segment trace = Segment{
			Point{(double)p, 0},
			Point{(double)p, (double)actual_sinogram->rows-1}
		};

		// Set-up the trace iterator
		TraceIterator<double> iterator(*actual_sinogram, trace);
		assert(iterator.valid());

		// Apply the functional
		double pixel = (*functional)(iterator);
		circus.at<double>(
			0,	// row
			p	// column
		) = pixel;
	}

	if (preprocessed)
		delete actual_sinogram;

	return circus;
}

#endif
