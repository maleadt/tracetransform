//
// Configuration
//

// System includes
#include <iostream>
#include <string>
#include <cmath>
#include <complex>
#include <iomanip>
#include <ctime>

// OpenCV includes
#include <cv.h>
#include <highgui.h>

// Local includes
#include "auxiliary.h"
#include "traceiterator.h"
#include "tracetransform.h"
#include "orthocircusfunction.h"

// Debug flags
#define DEBUG_IMAGES


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
// Auxiliary
//

struct Profiler
{
	Profiler()
	{
		t1 = clock();
	}

	double elapsed()
	{
		t2 = clock();
		return (double)(t2-t1)/CLOCKS_PER_SEC;
	}

	time_t t1, t2;
};



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
	nullptr,
	pfunctional_1
};

int main(int argc, char **argv)
{
	// Check and read the parameters
	if (argc < 4) {
		std::cerr << "Invalid usage: " << argv[0] << " INPUT T-FUNCTIONALS P-FUNCTIONALS" << std::endl;
		return 1;
	}
	std::string fn_input = argv[1];

	// Get the chosen T-functionals
	std::stringstream ss;
	ss << argv[2];
	unsigned short i;
	std::vector<unsigned short> chosen_tfunctionals;
	while (ss >> i) {
		if (ss.fail() || i >= TFUNCTIONALS.size() || TFUNCTIONALS[i] == nullptr) {
			std::cerr << "Error: invalid T-functional provided" << std::endl;
			return 1;
		}
		chosen_tfunctionals.push_back(i);
		if (ss.peek() == ',')
			ss.ignore();
	}

	// Get the chosen P-functional
	ss.clear();
	ss << argv[3];
	std::vector<unsigned short> chosen_pfunctionals;
	while (ss >> i) {
		if (ss.fail() || i >= PFUNCTIONALS.size() || PFUNCTIONALS[i] == nullptr) {
			std::cerr << "Error: invalid P-functional provided" << std::endl;
			return 1;
		}
		chosen_pfunctionals.push_back(i);
		if (ss.peek() == ',')
			ss.ignore();
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

	// Save profiling data
	std::vector<double> truntimes(chosen_tfunctionals.size());
	std::vector<double> pruntimes(chosen_tfunctionals.size());

	// Process all T-functionals
	cv::Mat data;
	int decimals = 7;	// size of the column header
	std::cerr << "Calculating ";
	for (size_t t = 0; t < chosen_tfunctionals.size(); t++) {
		// Calculate the trace transform sinogram
		std::cerr << " T" << chosen_tfunctionals[t] << "..." << std::flush;
		Profiler tprofiler;
		cv::Mat sinogram = getTraceTransform(
			input,
			1,	// angle resolution
			1,	// distance resolution
			TFUNCTIONALS[chosen_tfunctionals[t]]
		);
		truntimes[t] = tprofiler.elapsed();

		// Show the trace transform sinogram
		std::stringstream sinogram_title;
		sinogram_title << "sinogram after functional T" << chosen_tfunctionals[t];
		#ifdef DEBUG_IMAGES
		cv::imshow(sinogram_title.str(), mat2gray(sinogram));
		#endif

		// Process all P-functionals
		for (size_t p = 0; p < chosen_pfunctionals.size(); p++) {
			// Calculate the circus function
			std::cerr << " P" << chosen_pfunctionals[p] << "..." << std::flush;
			Profiler pprofiler;
			cv::Mat circus = getOrthonormalCircusFunction(
				sinogram,
				PFUNCTIONALS[chosen_pfunctionals[p]]
			);
			pruntimes[p] = pprofiler.elapsed();

			// Allocate the data
			if (data.empty()) {
				data = cv::Mat(
					cv::Size(
						 circus.cols,
						 chosen_tfunctionals.size()*chosen_pfunctionals.size()
					),
					CV_64FC1
				);
			} else {
				assert(data.cols == circus.cols);
			}

			// Copy the data
			for (int i = 0; i < circus.cols; i++) {
				double pixel = circus.at<double>(0, i);
				data.at<double>(
					t+p*chosen_pfunctionals.size(),	// row
					i				// column
				) = pixel;
				decimals = std::max(decimals, (int)std::log10(pixel)+3);
			}
		}
	}
	std::cerr << std::endl;

	// Output the headers
	decimals += 2;
	std::cout << std::setiosflags(std::ios::fixed)
		<< std::setprecision(2)
		<< std::left;
	std::cout << "#  ";
	for (size_t tp = 0; tp < (unsigned)data.rows; tp++) {
		size_t p = tp % chosen_pfunctionals.size();
		size_t t = tp / chosen_pfunctionals.size();
		std::stringstream header;
		header << "T" << chosen_tfunctionals[t]
			<< "-P" << chosen_pfunctionals[p];
		std::cout << std::setw(decimals) << header.str();
	}
	std::cout << "\n";

	// Output the data
	for (int i = 0; i < data.cols; i++) {
		std::cout << "   ";
		for (int tp = 0; tp < data.rows; tp++) {
			std::cout << std::setw(decimals)
				<< data.at<double>(tp, i);
		}
		std::cout << "\n";
	}
	std::cout << std::flush;

	// Output the runtimes
	std::cerr << std::setiosflags(std::ios::fixed) << std::setprecision(0);
	std::cerr << "Runtime of T-functionals:\n";
	for (size_t t = 0; t < chosen_tfunctionals.size(); t++) {
		std::cerr << "   T" << chosen_tfunctionals[t]<< ": "
			<< 1000*truntimes[t] << " ms\n";
	}
	std::cerr << "Runtime of P-functionals:\n";
	for (size_t t = 0; t < chosen_pfunctionals.size(); t++) {
		std::cerr << "   P" << chosen_pfunctionals[t] << ": "
			<< 1000*pruntimes[t]/(double)chosen_tfunctionals.size()
			<< " ms\n";
	}
	std::cerr << std::flush;

	// Give the user time to look at the images
	#ifdef DEBUG_IMAGES
	cv::waitKey();
	#endif

	return 0;
}
