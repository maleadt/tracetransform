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
#include "iterators.h"


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
			origin.x - std::cos(a_rad) * diagonal/2.0,
			origin.y - std::sin(a_rad) * diagonal/2.0
		}, Point{
			origin.x + std::cos(a_rad) * diagonal/2.0,
			origin.y + std::sin(a_rad) * diagonal/2.0
		}
	};
}

cv::Mat getTraceTransform(
	const cv::Mat &image,
	const double a_stepsize,
	const double p_stepsize,
	TFunctional<double, double> *functional)
{
	assert(a_stepsize > 0);
	assert(p_stepsize > 0);
	assert(image.type() == CV_64FC1);

	// Calculate and create the transform matrix
	double diagonal = std::hypot(image.size().width, image.size().height);
	unsigned int a_steps = (unsigned int) std::ceil(360.0 / a_stepsize);
	unsigned int p_steps = (unsigned int) std::ceil(diagonal / p_stepsize);
	cv::Mat transform = cv::Mat::zeros(
		(int) p_steps,	// rows
		(int) a_steps,	// columns
		CV_64FC1);

	// Process all angles
	for (unsigned int a_step = 0; a_step < a_steps; a_step++) {
		double a = a_step * a_stepsize;

		// Calculate projection line
		Segment proj = bounding_segment(
			image.size(),
			a,
			Point{image.size().width/2.0, image.size().height/2.0}
		);

		// Partition the projection line in projection bands
		for (unsigned int p_step = 0; p_step < p_steps; p_step++) {
			double p = p_step * p_stepsize;
			double proj_x = proj.begin.x + p*proj.rcx();
			double proj_y = proj.begin.y + p*proj.rcy();

			// Determine the trace segment
			Segment trace = bounding_segment(
				image.size(),
				a + 90,
				Point{proj_x, proj_y}
			);

			// Set-up the trace iterator
			LineIterator<double> iterator(image, trace);

			// Apply the functional
			if (iterator.valid()) {
				double pixel = (*functional)(&iterator);
				transform.at<double>(
					(int) p_step,	// row
					(int) a_step	// column
				) = pixel;
			}
		}
	}

	return transform;
}

#endif
