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

// Local includes
#include "auxiliary.h"


//
// Routines
//

// Calculate the endpoints of the bounding segment
Segment bounding_segment(const cv::Size &size,
	const double a,
	const Point origin)
{
	double a_rad = deg2rad(a);

	double diagonal = std::hypot(size.width, size.height);

	return Segment{
		Point{
			origin.x - std::cos(a_rad) * diagonal/2,
			origin.y - std::sin(a_rad) * diagonal/2
		}, Point{
			origin.x + std::cos(a_rad) * diagonal/2,
			origin.y + std::sin(a_rad) * diagonal/2
		}
	};
}

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
	Point origin{std::floor(input.size().width/2), std::floor(input.size().height/2)};

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
		cv::Mat transform = cv::getRotationMatrix2D(origin.getPoint2f(), a+90, 1.0);
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
