//
// Configuration
//

// System includes
#include <iostream>
#include <string>
#include <complex>
#include <iomanip>
#include <ctime>

// OpenCV includes
#include <cv.h>
#include <highgui.h>

// Local includes
#include "auxiliary.h"
#include "traceiterator.h"
#include "functionals.h"
#include "tracetransform.h"
#include "orthocircusfunction.h"

// Debug flags
//#define DEBUG_IMAGES


//
// Auxiliary
//

struct Profiler
{
	Profiler()
	{
		t1 = clock();
	}

	void stop()
	{
		t2 = clock();		
	}

	double elapsed()
	{
		return (double)(t2-t1)/CLOCKS_PER_SEC;
	}

	time_t t1, t2;
};



//
// Main
//

// Available T-functionals
const std::vector<TFunctional<uchar,double>*> TFUNCTIONALS{
	new TFunctionalRadon<uchar>(),
	new TFunctional1<uchar>(),
	new TFunctional2<uchar>(),
	new TFunctional3<uchar>(),
	new TFunctional4<uchar>(),
	new TFunctional5<uchar>()
};

// Available P-functionals
const std::vector<PFunctional<double,double>*> PFUNCTIONALS{
	nullptr,
	new PFunctional1<double>(),
	new PFunctional2<double>(),
	new PFunctional3<double>()
	// SIZE = Hermite functional
};
const unsigned short PFUNCTIONAL_HERMITE = PFUNCTIONALS.size();
enum PFunctionalType {
	REGULAR,
	HERMITE
};

int main(int argc, char **argv)
{
	// Check and read the parameters
	if (argc < 4) {
		std::cerr << "Invalid usage: " << argv[0]
			<< " INPUT T-FUNCTIONALS P-FUNCTIONALS" << std::endl;
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
	std::vector<unsigned short> chosen_pfunctionals_parameter;
	while (!ss.eof()) {
		PFunctionalType type = REGULAR;
		if (ss.peek() == 'H') {
			type = HERMITE;
			ss.ignore();
			ss >> i;
		} else {
			ss >> i;
		}

		if (ss.fail() || i >= PFUNCTIONALS.size() || PFUNCTIONALS[i] == nullptr) {
			std::cerr << "Error: invalid P-functional provided" << std::endl;
			return 1;
		}

		switch (type) {
			case REGULAR:
				chosen_pfunctionals.push_back(i);
				chosen_pfunctionals_parameter.push_back(0);
				break;
			case HERMITE:
				chosen_pfunctionals.push_back(PFUNCTIONAL_HERMITE);
				chosen_pfunctionals_parameter.push_back(i);
				break;
		}

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
	std::vector<double> runtimes(chosen_tfunctionals.size()*chosen_pfunctionals.size());

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
		tprofiler.stop();

		// Show the trace transform sinogram
		std::stringstream sinogram_title;
		sinogram_title << "sinogram after functional T" << chosen_tfunctionals[t];
		#ifdef DEBUG_IMAGES
		cv::imshow(sinogram_title.str(), mat2gray(sinogram));
		#endif

		// Process all P-functionals
		for (size_t p = 0; p < chosen_pfunctionals.size(); p++) {
			// Calculate the circus function
			Profiler pprofiler;
			cv::Mat circus;
			if (chosen_pfunctionals[p] == PFUNCTIONAL_HERMITE) {
				std::cerr << " H" << chosen_pfunctionals_parameter[p] << "..." << std::flush;
				circus = getOrthonormalCircusFunction(
					sinogram,
					chosen_pfunctionals_parameter[p]
				);
			} else {
				std::cerr << " P" << chosen_pfunctionals[p] << "..." << std::flush;
				circus = getCircusFunction(
					sinogram,
					PFUNCTIONALS[chosen_pfunctionals[p]]
				);
			}
			pprofiler.stop();
			runtimes[t*chosen_pfunctionals.size()+p]
				= tprofiler.elapsed() + pprofiler.elapsed();

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
					t*chosen_pfunctionals.size()+p,	// row
					i				// column
				) = pixel;
				decimals = std::max(decimals, (int)std::log10(pixel)+3);
			}
		}
	}
	std::cerr << std::endl;

	// Output the headers
	decimals += 2;
	std::cout << "#  ";
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(0);
	for (size_t tp = 0; tp < (unsigned)data.rows; tp++) {
		size_t t = tp / chosen_pfunctionals.size();
		size_t p = tp % chosen_pfunctionals.size();
		std::stringstream header;
		header << "T" << chosen_tfunctionals[t];
		if (chosen_pfunctionals[p] == PFUNCTIONAL_HERMITE)
			header << "-H" << chosen_pfunctionals_parameter[p];
		else
			header << "-P" << chosen_pfunctionals[p];
		std::cout << std::setw(decimals) << header.str();
	}
	std::cout << "\n";

	// Output the data
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2);
	for (int i = 0; i < data.cols; i++) {
		std::cout << "   ";
		for (int tp = 0; tp < data.rows; tp++) {
			std::cout << std::setw(decimals)
				<< data.at<double>(tp, i);
		}
		std::cout << "\n";
	}

	// Output the footer
	std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(0);
	std::cout << "#  ";
	for (size_t tp = 0; tp < (unsigned)data.rows; tp++) {
		std::stringstream runtime;
		runtime << 1000.0*runtimes[tp] << "ms";
		std::cout << std::setw(decimals) << runtime.str();
	}
	std::cout << std::endl;

	// Give the user time to look at the images
	#ifdef DEBUG_IMAGES
	cv::waitKey();
	#endif

	return 0;
}
