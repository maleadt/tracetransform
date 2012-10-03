//
// Configuration
//

// Include guard
#ifndef AUXILIARY_H
#define AUXILIARY_H

// System includes
#include <iostream>
#include <fstream>
#include <sstream>

// Library includes
#include <cv.h>
#include <Eigen/Dense>


//
// Data
//


//
// Structs
//

struct Point
{
	Point operator*(const double factor) const {
		return Point{x*factor, y*factor};
	}

	Point operator+(const Point& term) const {
		return Point{x+term.x, y+term.y};
	}

	Point operator-(const Point& term) const {
		return Point{x-term.x, y-term.y};
	}

	double x;
	double y;
};

std::ostream& operator<<(std::ostream &stream, const Point& point) {
	stream << point.x << "x" << point.y;
	return stream;
}



//
// Routines
//

// Temporary conversion routine to ease migration from OpenCV to Eigen3
cv::Mat eigen2opencv(const Eigen::MatrixXd &eigen) {
	cv::Mat opencv(eigen.rows(), eigen.cols(), CV_64FC1);
	for (size_t i = 0; i < eigen.rows(); i++) {
		for (size_t j = 0; j < eigen.cols(); j++) {
			opencv.at<double>(i, j) = eigen(i, j);
		}
	}
	return opencv;
}

// Read an ASCII PGM file
Eigen::MatrixXd readPgm(std::string filename)
{
	std::ifstream infile(filename);
	std::string inputLine = "";

	// First line: version
	getline(infile, inputLine);
	if (inputLine.compare("P2") != 0) {
		std::cerr << "readPGM: invalid PGM version " << inputLine << std::endl;
		exit(1);
	}

	// Second line: comment (optional)
	if (infile.peek() == '#')
		getline(infile, inputLine);

	// Continue with a stringstream
	std::stringstream ss;
	ss << infile.rdbuf();

	// Size
	unsigned int numrows = 0, numcols = 0;
	ss >> numcols >> numrows;
	Eigen::MatrixXd data(numrows, numcols);

	// Maxval
	unsigned int maxval;
	ss >> maxval;
	assert(maxval == 255);

	// Data
	double value;
	for (unsigned int row = 0; row < numrows; row++) {
		for (unsigned int col = 0; col < numcols; col++) {
			ss >> value;
			data(row, col) = value;
		}
	}
	infile.close();

	return data;
}

// Write an ASCII PGM file
void writePgm(const Eigen::MatrixXd &data, std::string filename)
{
	std::ofstream outfile(filename);

	// First line: version
	outfile << "P2" << "\n";

	// Second line: size
	outfile << data.cols() << " " << data.rows() << "\n";

	// Third line: maxval
	outfile << 255 << "\n";

	// Data
	long pos = outfile.tellp();
	for (unsigned int row = 0; row < data.rows(); row++) {
		for (unsigned int col = 0; col < data.cols(); col++) {
			outfile << data(row, col);
			if (outfile.tellp() - pos > 66) {
				outfile << "\n";
				pos = outfile.tellp();
			} else {
				outfile << " ";
			}
		}
	}
	outfile.close();
}

// Convert a grayscale image (range [0, 255]) to a matrix (range [0, 1]).
Eigen::MatrixXd gray2mat(const Eigen::MatrixXd &grayscale)
{
	Eigen::MatrixXd matrix(grayscale.rows(), grayscale.cols());
	// TODO: value scale intrinsic?
	for (unsigned int row = 0; row < matrix.rows(); row++) {
		for (unsigned int col = 0; col < matrix.cols(); col++) {
			matrix(row, col) = grayscale(row, col) / 255.0;
		}
	}
	return matrix;
}

// Convert a matrix (arbitrary values) to a grayscale image (range [0, 255]). This
// involves detecting the maximum value, and clamping that to 255.
template <typename T>
cv::Mat mat2gray(const cv::Mat &matrix)
{
	T maximum = 0;
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			T pixel = matrix.at<T>(i, j);
			if (pixel > maximum)
				maximum = pixel;
		}
	}
	cv::Mat grayscale(matrix.size(), CV_8UC1);
	matrix.convertTo(grayscale, CV_8UC1, 255.0/maximum, 0);
	return grayscale;
}

template <typename T>
double arithmetic_mean(const cv::Mat &vector)
{
	assert(vector.rows == 1);
	if (vector.cols <= 0)
		return NAN;

	double sum = 0;
	for (int i = 0; i < vector.cols; i++) {
		sum += vector.at<T>(0, i);
	}

	return sum / vector.cols;
}

template <typename T>
double standard_deviation(const cv::Mat &vector)
{
	assert(vector.rows == 1);
	if (vector.cols <= 0)
		return NAN;

	double mean = arithmetic_mean<T>(vector);
	double sum = 0;
	for (int i = 0; i < vector.cols; i++) {
		double diff = vector.at<T>(0, i) - mean;
		sum += diff*diff;
	}
	
	// NOTE: this is the default MATLAB interpretation, Wiki's std()
	//       uses vector.cols
	return std::sqrt(sum / (vector.cols-1));
}

template <typename T>
cv::Mat zscore(const cv::Mat &vector)
{
	assert(vector.rows == 1);
	if (vector.cols <= 0)
		return cv::Mat();

	double mean = arithmetic_mean<T>(vector);
	double stdev = standard_deviation<T>(vector);

	cv::Mat transformed(vector.size(), vector.type());
	for (int i = 0; i < vector.cols; i++) {
		transformed.at<T>(0, i) = (vector.at<T>(0, i) - mean) / stdev;
	}

	return transformed;
}

#endif
