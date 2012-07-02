//
// Configuration
//

// System includes
#include <iostream>
#include <string>

// OpenCV includes
#include <cv.h>
#include <highgui.h>

// Local includes
#include "auxiliary.h"
#include "tracetransform.h"
#include "traceiterator.h"


//
// Iterator helpers
//

unsigned long iterator_sum(TraceIterator &iterator)
{
	unsigned long sum = 0;
	while (iterator.hasNext()) {
		sum += iterator.value();
		iterator.next();
	}
	return sum;
}

Point iterator_weighedmedian(TraceIterator &iterator)
{
	unsigned long sum = iterator_sum(iterator);
	iterator.toFront();

	unsigned long cumsum = 0;
	Point median;
	while (iterator.hasNext()) {
		median = iterator.point();
		cumsum += iterator.value();
		iterator.next();

		if (cumsum > sum/2.0) {
			break;
		}
	}
	return median;
}


//
// T functionals
//

// T-functional for the Radon transform.
//
// T(f(t)) = Int[0-inf] f(t)dt
double tfunctional_radon(TraceIterator &iterator)
{
	return (double) iterator_sum(iterator);
}

// T(f(t)) = Int[0-inf] t*f(t)dt
double tfunctional_1_kernel(TraceIterator &iterator)
{
	unsigned long sum = 0;
	for (unsigned int t = 0; iterator.hasNext(); t++) {
		sum += iterator.value() * t;
		iterator.next();
	}
	return (double) sum;
}

// T(f(t)) = Int[0-inf] r*f(r)dr
double tfunctional_1(TraceIterator &iterator)
{
	// Transform the domain from t to r, and integrate
	Point median = iterator_weighedmedian(iterator);
	TraceIterator iterator_positive = iterator.transformDomain(
		Segment(
			median,
			iterator.segment().end
		)
	);

	// TODO: directly transform the negative domain as well?
	/*
	TraceIterator iterator_negative = iterator.transformDomain(
		Segment(
			median,
			iterator.segment().begin
		)
	);
	*/
	return tfunctional_1_kernel(iterator_positive);
}

// T(f(t)) = Int[0-inf] t^2*f(t)dt
double tfunctional_2_kernel(TraceIterator &iterator)
{
	unsigned long sum = 0;
	for (unsigned int t = 0; iterator.hasNext(); t++) {
		sum += iterator.value() * t*t;
		iterator.next();
	}
	return (double) sum;
}

// T(f(t)) = Int[0-inf] r^2*f(r)dr
double tfunctional_2(TraceIterator &iterator)
{
	// Transform the domain from t to r, and integrate
	Point median = iterator_weighedmedian(iterator);
	TraceIterator iterator_positive = iterator.transformDomain(
		Segment(
			median,
			iterator.segment().end
		)
	);
	return tfunctional_2_kernel(iterator_positive);
}


//
// Main
//

int main(int argc, char **argv)
{
	// Check and read the parameters
	if (argc < 2) {
		std::cerr << "Invalid usage: " << argv[0] << " INPUT [OUTPUT]" << std::endl;
		return 1;
	}
	std::string fn_input = argv[1];

	// Read the image
	cv::Mat input = cv::imread(
		fn_input,	// filename
		0		// flags (0 = force grayscale)
	);
	if (input.empty()) {
		std::cerr << "Error: could not load image" << std::endl;
		return 1;
	}

	// Get the trace transform
	cv::Mat transform = getTraceTransform(
		input,
		1,	// angle resolution
		1,	// distance resolution
		tfunctional_2
	);

	// Scale the transform back to the [0,255] intensity range
	double maximum = 0;
	for (int i = 0; i < transform.rows; i++) {
		for (int j = 0; j < transform.cols; j++) {
			double pixel = transform.at<double>(i, j);
			if (pixel > maximum)
				maximum = pixel;
		}
	}
	cv::Mat transform_scaled(transform.size(), CV_8UC1);
	transform.convertTo(transform_scaled, CV_8UC1, 255.0/maximum, 0);

	// Display or write the image
	if (argc < 3) {
		cv::imshow("Trace transform", (transform_scaled));
		cv::waitKey();
	} else {
		std::string fn_output = argv[2];
		cv::imwrite(
			fn_output,
			transform_scaled
		);
	}

	return 0;
}
