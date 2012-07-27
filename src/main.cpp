//
// Configuration
//

// System includes
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>

// OpenCV includes
#include <cv.h>
#include <highgui.h>

// Local includes
#include "auxiliary.h"
#include "functionals.h"
#include "tracetransform.h"
#include "circusfunction.h"

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

std::string ordinalSuffix(unsigned long n) {
	// Numbers from 11 to 13 don't have st, nd, rd
	if (10 < (n%100) && (n%100) < 14)
		return "th";

	switch(n % 10) {
	case 1:
		return "st";

	case 2:
		return "nd";

	case 3:
		return "rd";

	default:
		return "th";
	}
} 


//
// Main
//

enum PFunctionalType {
       REGULAR,
       HERMITE
};

int main(int argc, char **argv)
{
	// Check and read the parameters
	if (argc < 3) {
		std::cerr << "Invalid usage: " << argv[0]
			<< " INPUT T-FUNCTIONALS [P-FUNCTIONALS]" << std::endl;
		return 1;
	}
	std::string fn_input = argv[1];

	// Get the chosen T-functionals
	std::vector<TFunctional<double,double>*> tfunctionals;
	std::stringstream ss;
	ss << argv[2];
	while (!ss.eof()) {
		unsigned short i;
		ss >> i;
		if (ss.fail()) {
			std::cerr << "Error: unparseable T-functional identifier" << std::endl;
			return 1;
		}

		switch (i) {
		case 0:
			tfunctionals.push_back(new TFunctionalRadon<double>());
			break;
		case 1:
			tfunctionals.push_back(new TFunctional1<double>());
			break;
		case 2:
			tfunctionals.push_back(new TFunctional2<double>());
			break;
		case 3:
			tfunctionals.push_back(new TFunctional3<double>());
			break;
		case 4:
			tfunctionals.push_back(new TFunctional4<double>());
			break;
		case 5:
			tfunctionals.push_back(new TFunctional5<double>());
			break;
		default:
			std::cerr << "Error: invalid T-functional provided" << std::endl;
			return 1;
		}

		if (ss.peek() == ',')
			ss.ignore();
	}

	// Get the chosen P-functional
	std::vector<PFunctional<double,double>*> pfunctionals;
	if (argc >= 4) {
		ss.clear();
		ss << argv[3];
		while (!ss.eof()) {
			PFunctionalType type = REGULAR;
			if (ss.peek() == 'H') {
				type = HERMITE;
				ss.ignore();
			}

			unsigned short i;
			ss >> i;
			if (ss.fail()) {
				std::cerr << "Error: unparseable P-functional identifier" << std::endl;
				return 1;
			}

			switch (type) {
			case REGULAR:
			{
				switch (i) {
				case 1:
					pfunctionals.push_back(new PFunctional1<double>());
					break;
				case 2:
					pfunctionals.push_back(new PFunctional2<double>());
					break;
				case 3:
					pfunctionals.push_back(new PFunctional3<double>());
					break;
				default:
					std::cerr << "Error: invalid P-functional provided" << std::endl;
					return 1;
				}
				break;
			}
			case HERMITE:
				pfunctionals.push_back(new PFunctionalHermite<double>(i));
				break;	
			}

			if (ss.peek() == ',')
				ss.ignore();
		}
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
	input = gray2mat(input);

	// Save profiling data
	std::vector<double> runtimes(tfunctionals.size()*pfunctionals.size());

	// Process all T-functionals
	cv::Mat data;
	int decimals = 7;	// size of the column header
	std::cerr << "Calculating ";
	for (unsigned int t = 0; t < tfunctionals.size(); t++) {
		// Calculate the trace transform sinogram
		std::cerr << t+1 << ordinalSuffix(t+1) << " T" << "..." << std::flush;
		Profiler tprofiler;
		cv::Mat sinogram = getTraceTransform(
			input,
			1,	// angle resolution
			1,	// distance resolution
			tfunctionals[t]
		);
		tprofiler.stop();

		// Show the trace transform sinogram
		#ifdef DEBUG_IMAGES
		std::stringstream sinogram_title;
		sinogram_title << "sinogram after " << t+1 << ordinalSuffix(t+1)
			<< " T-functional";
		cv::imshow(sinogram_title.str(), mat2gray(sinogram));
		#endif

		// Process all P-functionals
		for (unsigned int p = 0; p < pfunctionals.size(); p++) {
			// Calculate the circus function
			std::cerr << p+1 << ordinalSuffix(p+1) << " P" << "..." << std::flush;
			Profiler pprofiler;
			cv::Mat circus = getCircusFunction(
				sinogram,
				pfunctionals[p]
			);
			pprofiler.stop();
			runtimes[t*pfunctionals.size()+p]
				= tprofiler.elapsed() + pprofiler.elapsed();

			// Allocate the data
			if (data.empty()) {
				data = cv::Mat(
					cv::Size(
						 circus.cols,
						 tfunctionals.size()*pfunctionals.size()
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
					t*pfunctionals.size()+p,	// row
					i				// column
				) = pixel;
				decimals = std::max(decimals, (int)std::log10(pixel)+3);
			}
		}
	}
	std::cerr << std::endl;

	// Display the results of the P-functionals
	if (pfunctionals.size() > 0) {
		// Headers
		decimals += 2;
		std::cout << "#  ";
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(0);
		for (size_t tp = 0; tp < (unsigned)data.rows; tp++) {
			size_t t = tp / pfunctionals.size();
			size_t p = tp % pfunctionals.size();
			std::stringstream header;
			header << t+1 << ordinalSuffix(t+1) << " T, "
				<< p+1 << ordinalSuffix(p+1) << " P";
			std::cout << std::setw(decimals) << header.str();
		}
		std::cout << "\n";

		// Data
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2);
		for (int i = 0; i < data.cols; i++) {
			std::cout << "   ";
			for (int tp = 0; tp < data.rows; tp++) {
				std::cout << std::setw(decimals)
					<< data.at<double>(tp, i);
			}
			std::cout << "\n";
		}

		// Footer
		std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(0);
		std::cout << "#  ";
		for (size_t tp = 0; tp < (unsigned)data.rows; tp++) {
			std::stringstream runtime;
			runtime << 1000.0*runtimes[tp] << "ms";
			std::cout << std::setw(decimals) << runtime.str();
		}
		std::cout << std::endl;
	}

	// Give the user time to look at the images
	#ifdef DEBUG_IMAGES
	cv::waitKey();
	#endif

	return 0;
}

int main2() {
	cv::Mat sinogram(
		6,
		4,
		CV_64FC1
	);

	sinogram.at<double>(0, 0) = 2;
	sinogram.at<double>(0, 1) = 1;
	sinogram.at<double>(0, 2) = 1;
	sinogram.at<double>(0, 3) = 8;

	sinogram.at<double>(1, 0) = 5;
	sinogram.at<double>(1, 1) = 2;
	sinogram.at<double>(1, 2) = 2;
	sinogram.at<double>(1, 3) = 6;

	sinogram.at<double>(2, 0) = 7;
	sinogram.at<double>(2, 1) = 4;
	sinogram.at<double>(2, 2) = 3;
	sinogram.at<double>(2, 3) = 4;

	sinogram.at<double>(3, 0) = 2;
	sinogram.at<double>(3, 1) = 9;
	sinogram.at<double>(3, 2) = 5;
	sinogram.at<double>(3, 3) = 3;

	sinogram.at<double>(4, 0) = 3;
	sinogram.at<double>(4, 1) = 6;
	sinogram.at<double>(4, 2) = 10;
	sinogram.at<double>(4, 3) = 2;

	sinogram.at<double>(5, 0) = 1;
	sinogram.at<double>(5, 1) = 2;
	sinogram.at<double>(5, 2) = 2;
	sinogram.at<double>(5, 3) = 1;

	for (int i = 0; i < sinogram.rows; i++) {
		for (int j = 0; j < sinogram.cols; j++) {
			std::cerr << sinogram.at<double>(i, j) << "\t";
		}
		std::cerr << "\n";
	}
	std::cerr << "---------------------------------------------------" << std::endl;

	cv::Mat circus = getCircusFunction(
		sinogram,
		new PFunctionalHermite<double>(1)
	);

	for (int i = 0; i < circus.rows; i++) {
		for (int j = 0; j < circus.cols; j++) {
			std::cerr << circus.at<double>(i, j) << "\t";
		}
		std::cerr << "\n";
	}
	std::cerr << std::flush;
}
