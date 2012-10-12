//
// Configuration
//

// System includes
#include <iostream>

// Library includes
#include <cv.h>
#include <highgui.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Local includes
#include "auxiliary.h"

// Algorithm parameters
#define ANGLE_INTERVAL		1
#define ANGLE			31.4


//
// Main
//

int main(int argc, char **argv) {	
	// Check and read the parameters
	if (argc < 1) {
		std::cerr << "Invalid usage: " << argv[0]
			<< " INPUT" << std::endl;
		return 1;
	}
	std::string fninput = argv[1];

	// Read the image
	Eigen::MatrixXd input_eigen = pgmRead(fninput);
	input_eigen = gray2mat(input_eigen);
	cv::Mat input_opencv = eigen2opencv(input_eigen);

	// Stretch the matrices
	int ndiag = (int) std::ceil(360.0/ANGLE_INTERVAL);
	int size = (int) std::ceil(ndiag/std::sqrt(2));
	input_eigen = resize(input_eigen, size, size);
	cv::imwrite("stretched_eigen.pgm", mat2gray<double>(eigen2opencv(input_eigen)));
	cv::resize(input_opencv, input_opencv, cv::Size(size, size));
	cv::imwrite("stretched_opencv.pgm", mat2gray<double>(input_opencv));

	// Rotate the matrices
	input_opencv = eigen2opencv(input_eigen);	// not to have other differences
	cv::Point2d origin_opencv{(input_eigen.cols()-1)/2.0, (input_eigen.rows()-1)/2.0};
	cv::Mat transform_opencv = cv::getRotationMatrix2D(origin_opencv, ANGLE, 1.0);
	cv::Mat input_rotated_opencv;
	cv::warpAffine(input_opencv, input_rotated_opencv, transform_opencv, input_opencv.size());
	cv::imwrite("rotated_opencv.pgm", mat2gray<double>(input_rotated_opencv));
	Point origin_eigen((input_eigen.cols()-1)/2.0, (input_eigen.rows()-1)/2.0);
	Eigen::MatrixXd input_rotated_eigen = rotate(input_eigen, origin_eigen, deg2rad(ANGLE));
	cv::imwrite("rotated_eigen.pgm", mat2gray<double>(eigen2opencv(input_rotated_eigen)));

	return 0;
}