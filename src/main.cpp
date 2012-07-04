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
#include "traceiterator.h"
#include "tracetransform.h"
#include "orthocircusfunction.h"


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

		if (2*integral >= sum)
			break;
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

		if (2*integral >= sum)
			break;
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
// Laguerre P-functionals
//

// P(g(p)) = Sum(k) abs(g(p+1) -g(p))
double pfunctional_1(TraceIterator &iterator)
{
	unsigned long sum = 0;
	double previous;
	if (iterator.hasNext()) {
		previous = iterator.value();
		iterator.next();
	}
	while (iterator.hasNext()) {
		double current = iterator.value();
		sum += std::abs(previous -current);
		previous = current;
		iterator.next();
	}
	return (double)sum;
}


//
// Main
//

// Available T-functionals
const std::vector<Functional> TFUNCTIONALS{
	tfunctional_radon,
	tfunctional_1,
	tfunctional_2,
	tfunctional_3,
	tfunctional_4,
	tfunctional_5
};

// Available P-functionals
const std::vector<Functional> PFUNCTIONALS{
	pfunctional_1
};

int main(int argc, char **argv)
{
	// Check and read the parameters
	if (argc < 4) {
		std::cerr << "Invalid usage: " << argv[0] << " INPUT T-FUNCTIONAL P-FUNCTIONAL" << std::endl;
		return 1;
	}
	std::string fn_input = argv[1];

	// Get the chosen T-functional
	std::stringstream ss;
	ss << argv[2];
	unsigned short tfunctional;
	ss >> tfunctional;
	if (ss.fail() || tfunctional >= TFUNCTIONALS.size()) {
		std::cerr << "Error: invalid T-functional provided" << std::endl;
		return 1;
	}

	// Get the chosen P-functional
	ss.clear();
	ss << argv[3];
	unsigned short pfunctional;
	ss >> pfunctional;
	pfunctional--;
	if (ss.fail() || pfunctional >= PFUNCTIONALS.size()) {
		std::cerr << "Error: invalid P-functional provided" << std::endl;
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

	// Calculate the trace transform sinogram
	cv::Mat sinogram = getTraceTransform(
		input,
		1,	// angle resolution
		1,	// distance resolution
		TFUNCTIONALS[tfunctional]
	);

	// Calculate the circus function
	cv::Mat circus = getOrthonormalCircusFunction(
		sinogram,
		PFUNCTIONALS[pfunctional]
	);

	// Return the output of the circus function
	for (int p = 0; p < circus.cols; p++) {
		std::cout << circus.at<double>(0, p) << std::endl;
	}

	return 0;
}
