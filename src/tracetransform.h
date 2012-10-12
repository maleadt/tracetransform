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
	const Eigen::MatrixXd &_input,
	const double a_stepsize,
	const double p_stepsize,
	Functional tfunctional,
	void *tfunctional_arguments)
{
	assert(a_stepsize > 0);
	assert(p_stepsize > 0);
	assert(_input.rows() == _input.cols());	// padded image!
	cv::Mat input = eigen2opencv(_input);

	// Get the image origin to rotate around
	cv::Point2d origin{(_input.cols()-1)/2.0, (_input.rows()-1)/2.0};

	// Calculate and allocate the output matrix
	unsigned int a_steps = (unsigned int) std::floor(360 / a_stepsize);
	unsigned int p_steps = (unsigned int) std::floor(_input.rows() / p_stepsize);
	cv::Mat output = cv::Mat::zeros(
		(int) p_steps,	// rows
		(int) a_steps,	// columns
		CV_64FC1);

	// Process all angles
	for (unsigned int a_step = 0; a_step < a_steps; a_step++) {
		// Calculate the transform matrix and rotate the image
		double a = a_step * a_stepsize;
		cv::Mat transform = cv::getRotationMatrix2D(origin, a+90, 1.0);
		cv::Mat _input_rotated;
		cv::warpAffine(input, _input_rotated, transform, input.size());

		// Process all projection bands
		for (unsigned int p_step = 0; p_step < p_steps; p_step++) {
			output.at<double>(
				(signed) (p_steps - p_step - 1),	// row
				(signed) a_step				// column
			) = tfunctional(
				_input_rotated.ptr<double>((int) (p_step*p_stepsize)),
				_input.cols(),
				tfunctional_arguments);
		}
	}

	return output;
}

#endif
