//
// Configuration
//

// System includes
#include <iostream>

// OpenCV includes
#include <cv.h>

// Local includes
#include "auxiliary.h"
extern "C" {
	#include "functionals.h"
}
#include "tracetransform.h"
#include "circusfunction.h"


//
// Main
//

int main() {
	//
	// Create data
	//

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

	std::cout << "Input sinogram:" << std::endl;

	for (int i = 0; i < sinogram.rows; i++) {
		std::cout << "\t";
		for (int j = 0; j < sinogram.cols; j++) {
			std::cout << sinogram.at<double>(i, j) << "\t";
		}
		std::cout << "\n";
	}

	std::cout << std::endl;


	//
	// Nearest orthonormal sinogram
	//

	// MATLAB: [Circus NOS] = Apply_Pfunct(data, 5, 1)
	//         (ajust Apply_Pfunct to return NOS as well)

	unsigned int center;
	cv::Mat nos = nearest_orthonormal_sinogram(sinogram, center);

	std::cout << "Nearest orthonormal sinogram (centered around row " << center << "):" << std::endl;

	for (int i = 0; i < nos.rows; i++) {
		std::cout << "\t";
		for (int j = 0; j < nos.cols; j++) {
			std::cout << nos.at<double>(i, j) << "\t";
		}
		std::cout << "\n";
	}

	std::cout << std::endl;


	//
	// Circus functions
	//

	// MATLAB: [Circus NOS] = Apply_Pfunct(data, 5, 1)

	std::cout << "Output of circus functions after Hermite P-functional (2nd order):" << std::endl;
	cv::Mat circus = getCircusFunction(
		sinogram,
		PFunctionalHermite,
		new ArgumentsHermite{2, 0}
	);

	for (int i = 0; i < circus.rows; i++) {
		std::cout << "\t";
		for (int j = 0; j < circus.cols; j++) {
			std::cout << circus.at<double>(i, j) << "\t";
		}
		std::cout << "\n";
	}
	std::cout << std::flush;
}

