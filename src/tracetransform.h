//
// Configuration
//

// Include guard
#ifndef TRACETRANSFORM_H
#define TRACETRANSFORM_H

// System includes
#include <limits>

// OpenCV includes
#include <cv.h>


//
// Routines
//

cv::Mat getTraceTransform(
	const cv::Mat &input,
	const double a_stepsize,
	const double p_stepsize,
	Functional tfunctional,
	void *tfunctional_arguments)
{
	assert(a_stepsize > 0);
	assert(p_stepsize > 0);
	assert(input.size().width == input.size().height);	// padded image!
	assert(input.type() == CV_64FC1);

	// Get the image origin to rotate around
	cv::Point2d origin{(input.size().width-1)/2.0, (input.size().height-1)/2.0};

	// Calculate and allocate the output matrix
	unsigned int a_steps = (unsigned int) std::floor(360 / a_stepsize);
	unsigned int p_steps = (unsigned int) std::floor(input.size().height / p_stepsize);
	cv::Mat output = cv::Mat::zeros(
		(int) p_steps,	// rows
		(int) a_steps,	// columns
		CV_64FC1);

	// Process all angles
	for (unsigned int a_step = 0; a_step < a_steps; a_step++) {
		// Calculate the transform matrix and rotate the image
		double a = a_step * a_stepsize;
		cv::Mat transform = cv::getRotationMatrix2D(origin, a+90, 1.0);
		cv::Mat input_rotated;
		cv::warpAffine(input, input_rotated, transform, input.size());

		// Process all projection bands
		for (unsigned int p_step = 0; p_step < p_steps; p_step++) {
			output.at<double>(
				(signed) (p_steps - p_step - 1),	// row
				(signed) a_step				// column
			) = tfunctional(
				input_rotated.ptr<double>((int) (p_step*p_stepsize)),
				input.cols,
				tfunctional_arguments);
		}
	}

	return output;
}

#endif
