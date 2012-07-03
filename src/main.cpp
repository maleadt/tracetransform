//
// Configuration
//

// System includes
#include <iostream>
#include <string>
#include <cmath>
#include <complex>

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

// Look for the median of the weighed indexes
//
// Conceptually this expands the list of indexes to a weighed one (in which each
// index is repeated as many times as the pixel value it represents), after
// which the median value of that array is located.
Point iterator_weighedmedian(TraceIterator &iterator)
{
	unsigned long sum = 0;
	while (iterator.hasNext()) {
		sum += iterator.value();
		iterator.next();
	}
	iterator.toFront();

	unsigned long integral = 0;
	Point median;
	while (iterator.hasNext()) {
		median = iterator.point();
		integral += iterator.value();
		iterator.next();

		if (integral > sum/2.0) {
			break;
		}
	}
	return median;
}

// Look for the median of the weighed indexes, but take the square root of the
// pixel values as weight
Point iterator_weighedmedian_sqrt(TraceIterator &iterator)
{
	double sum = 0;
	while (iterator.hasNext()) {
		sum += std::sqrt(iterator.value());
		iterator.next();
	}
	iterator.toFront();

	double integral = 0;
	Point median;
	while (iterator.hasNext()) {
		median = iterator.point();
		integral += std::sqrt(iterator.value());
		iterator.next();

		if (integral > sum/2.0) {
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
	double integral = 0;
	while (iterator.hasNext()) {
		integral += iterator.value();
		iterator.next();
	}
	return integral;
}

// T(f(t)) = Int[0-inf] r*f(r)dr
double tfunctional_1(TraceIterator &iterator)
{
	// Transform the domain from t to r
	Point r = iterator_weighedmedian(iterator);
	TraceIterator transformed = iterator.transformDomain(
		Segment{
			r,
			iterator.segment().end
		}
	);

	// Integrate
	double integral = 0;
	for (unsigned int t = 0; transformed.hasNext(); t++) {
		integral += transformed.value() * t;
		transformed.next();
	}
	return integral;
}

// T(f(t)) = Int[0-inf] r^2*f(r)dr
double tfunctional_2(TraceIterator &iterator)
{
	// Transform the domain from t to r
	Point r = iterator_weighedmedian(iterator);
	TraceIterator transformed = iterator.transformDomain(
		Segment{
			r,
			iterator.segment().end
		}
	);

	// Integrate
	double integral = 0;
	for (unsigned int t = 0; transformed.hasNext(); t++) {
		integral += transformed.value() * t*t;
		transformed.next();
	}
	return integral;
}

// T(f(t)) = Int[0-inf] exp(5i*log(r1))*r1*f(r1)dr1
double tfunctional_3(TraceIterator &iterator)
{
	// Transform the domain from t to r1
	Point r1 = iterator_weighedmedian_sqrt(iterator);
	TraceIterator transformed = iterator.transformDomain(
		Segment{
			r1,
			iterator.segment().end
		}
	);

	// Integrate
	std::complex<double> integral(0, 0);
	const std::complex<double> factor(0, 5);
	for (unsigned int t = 0; transformed.hasNext(); t++) {
		if (t > 0)	// since exp(i*log(0)) == 0
			integral += exp(factor*std::log(t))
				* (t*(double)transformed.value());
		transformed.next();
	}
	return std::abs(integral);
}

// T(f(t)) = Int[0-inf] exp(3i*log(r1))*f(r1)dr1
double tfunctional_4(TraceIterator &iterator)
{
	// Transform the domain from t to r1
	Point r1 = iterator_weighedmedian_sqrt(iterator);
	TraceIterator transformed = iterator.transformDomain(
		Segment{
			r1,
			iterator.segment().end
		}
	);

	// Integrate
	std::complex<double> integral(0, 0);
	const std::complex<double> factor(0, 3);
	for (unsigned int t = 0; transformed.hasNext(); t++) {
		if (t > 0)	// since exp(i*log(0)) == 0
			integral += exp(factor*std::log(t))
				* (double)transformed.value();
		transformed.next();
	}
	return std::abs(integral);
}

// T(f(t)) = Int[0-inf] exp(4i*log(r1))*sqrt(r1)*f(r1)dr1
double tfunctional_5(TraceIterator &iterator)
{
	// Transform the domain from t to r1
	Point r1 = iterator_weighedmedian_sqrt(iterator);
	TraceIterator transformed = iterator.transformDomain(
		Segment{
			r1,
			iterator.segment().end
		}
	);

	// Integrate
	std::complex<double> integral(0, 0);
	const std::complex<double> factor(0, 4);
	for (unsigned int t = 0; transformed.hasNext(); t++) {
		if (t > 0)	// since exp(i*log(0)) == 0
			integral += exp(factor*std::log(t))
				* (std::sqrt(t)*(double)transformed.value());
		transformed.next();
	}
	return std::abs(integral);
}


//
// Main
//


// Available T-functionals
const std::vector<TFunctional> TFUNCTIONALS{
	tfunctional_radon,
	tfunctional_1,
	tfunctional_2,
	tfunctional_3,
	tfunctional_4,
	tfunctional_5
};

int main(int argc, char **argv)
{
	// Check and read the parameters
	if (argc < 3) {
		std::cerr << "Invalid usage: " << argv[0] << " INPUT T-FUNCTIONAL [OUTPUT]" << std::endl;
		return 1;
	}
	std::string fn_input = argv[1];

	// Get the chosen functional
	std::stringstream ss(argv[2]);
	unsigned short tfunctional;
	ss >> tfunctional;
	if (ss.fail() || tfunctional >= TFUNCTIONALS.size()) {
		std::cerr << "Error: invalid T-functional provided" << std::endl;
		return 1;
	}

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
		TFUNCTIONALS[tfunctional]
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
	if (argc < 4) {
		cv::imshow("Trace transform", (transform_scaled));
		cv::waitKey();
	} else {
		std::string fn_output = argv[3];
		cv::imwrite(
			fn_output,
			transform_scaled
		);
	}

	return 0;
}
