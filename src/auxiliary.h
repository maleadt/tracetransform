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

typedef Eigen::RowVector2d Point;

std::ostream& operator<<(std::ostream &stream, const Point& point) {
	stream << point.x() << "x" << point.y();
	return stream;
}



//
// Routines
//

// Temporary conversion routines to ease migration from OpenCV to Eigen3
cv::Mat eigen2opencv(const Eigen::MatrixXd &eigen)
{
	cv::Mat opencv(eigen.rows(), eigen.cols(), CV_64FC1);
	for (size_t row = 0; row < eigen.rows(); row++) {
		for (size_t col = 0; col < eigen.cols(); col++) {
			opencv.at<double>(row, col) = eigen(row, col);
		}
	}
	return opencv;
}
Eigen::MatrixXd opencv2eigen(const cv::Mat &opencv)
{
	Eigen::MatrixXd eigen(opencv.rows, opencv.cols);
	for (int row = 0; row < opencv.rows; row++) {
		for (int col = 0; col < opencv.cols; col++) {
			eigen(row, col) = opencv.at<double>(row, col);
		}
	}
	return eigen;
}

// Read an ASCII PGM file
Eigen::MatrixXd pgmRead(std::string filename)
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
void pgmWrite(std::string filename, const Eigen::MatrixXd &data)
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
			outfile << (unsigned int) data(row, col);
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

// Write a MATLAB and gnuplot-compatible data file
void dataWrite(std::string filename, const Eigen::MatrixXd &data,
	const std::vector<std::string> &headers = std::vector<std::string>())
{
	assert(headers.size() == 0 || headers.size() == data.cols());

	// Calculate column width
	std::vector<unsigned int> widths(data.cols(), 0);
	for (size_t col = 0; col < data.cols(); col++) {
		if (headers.size() > 0)
			widths[col] = headers[col].length();
		for (size_t row = 0; row < data.rows(); row++) {
			double value = data(row, col);
			unsigned int width = 3;	// decimal, comma, 2 decimals
			if (value > 1)
				width += std::floor(std::log10(value));
			if (value < 0)	// dash for negative numbers
				width++;
			if (width > widths[col])
				widths[col] = width;
		}
		widths[col] += 2;	// add spacing
	}

	// Open file
	std::ofstream fd_data(filename);

	// Print headers
	if (headers.size() > 0) {
		fd_data << "%  ";
		fd_data << std::setiosflags(std::ios::fixed)
			<< std::setprecision(0);
		for (size_t col = 0; col < headers.size(); col++) {
			fd_data << std::setw(widths[col]) << headers[col];
		}
		fd_data << "\n";
	}

	// Print data
	fd_data << std::setiosflags(std::ios::fixed) << std::setprecision(2);
	for (size_t row = 0; row < data.rows(); row++) {
		fd_data << "   ";
		for (size_t col = 0; col < data.cols(); col++) {
			fd_data << std::setw(widths[col])
				<< data(row, col);
		}
		fd_data << "\n";
	}

	fd_data << std::flush;
	fd_data.close();
}

// Convert a grayscale image (range [0, 255]) to a matrix (range [0, 1]).
Eigen::MatrixXd gray2mat(const Eigen::MatrixXd &input)
{
	// Scale
	Eigen::MatrixXd output(input.rows(), input.cols());
	for (unsigned int col = 0; col < output.cols(); col++) {
		for (unsigned int row = 0; row < output.rows(); row++) {
			output(row, col) = input(row, col) / 255.0;
		}
	}
	return output;
}

// Convert a matrix (arbitrary values) to a grayscale image (range [0, 255]). This
// involves detecting the maximum value, and clamping that to 255.
Eigen::MatrixXd mat2gray(const Eigen::MatrixXd &input)
{
	// Detect maximum
	double maximum = 0;
	for (unsigned int col = 0; col < input.cols(); col++) {
		for (unsigned int row = 0; row < input.rows(); row++) {
			double pixel = input(row, col);
			if (pixel > maximum)
				maximum = pixel;
		}
	}

	// Scale
	Eigen::MatrixXd output(input.rows(), input.cols());
	for (unsigned int col = 0; col < output.cols(); col++) {
		for (unsigned int row = 0; row < output.rows(); row++) {
			output(row, col) = input(row, col) * 255.0/maximum;
		}
	}
	return output;
}

inline double deg2rad(double degrees)
{
	return (degrees * M_PI / 180);
}

double interpolate(const Eigen::MatrixXd &source, const Point &p)
{
	assert(p.x() >= 0 && p.x() <= source.cols()-1);
	assert(p.y() >= 0 && p.y() <= source.rows()-1);

	// Get fractional and integral part of the coordinates
	double x_int, y_int;
	double x_fract = std::modf(p.x(), &x_int);
	double y_fract = std::modf(p.y(), &y_int);

	return	  source((int)y_int, (int)x_int)*(1-x_fract)*(1-y_fract)
		+ source((int)y_int, (int)x_int+1)*x_fract*(1-y_fract)
		+ source((int)y_int+1, (int)x_int)*(1-x_fract)*y_fract
		+ source((int)y_int+1, (int)x_int+1)*x_fract*y_fract;

}

Eigen::MatrixXd resize(const Eigen::MatrixXd &input, const unsigned int rows, const unsigned int cols)
{	
	// Calculate transform matrix
	// TODO: use Eigen::Geometry
	Eigen::Matrix2d transform;
	transform <<	((double) input.rows()) / rows, 0,
			0, (((double) input.cols()) / cols);

	// Allocate output matrix
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows, cols);
	
	// Process all points
	// FIXME: borders are wrong (but this doesn't matter here since we
	//        only handle padded images)
	for (unsigned int col = 1; col < cols-1; col++) {
		for (unsigned int row = 1; row < rows-1; row++) {
			Point p(col, row);
			p += Eigen::RowVector2d(0.5, 0.5);
			p *= transform;
			p -= Eigen::RowVector2d(0.5, 0.5);
			output(row, col) = interpolate(input, p);
		}
	}
	return output;
}

Eigen::MatrixXd rotate(const Eigen::MatrixXd &input, const Point &origin, const double angle)
{	
	// Calculate transform matrix
	// TODO: use Eigen::Geometry
	// TODO: -angle?
	Eigen::Matrix2d transform;
	transform <<	std::cos(-angle), -std::sin(-angle),
			std::sin(-angle), std::cos(-angle);

	// Allocate output matrix
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(input.rows(), input.cols());
	
	// Process all points
	for (unsigned int col = 0; col < input.cols(); col++) {
		for (unsigned int row = 0; row < input.rows(); row++) {
			Point p(col, row);
			p -= origin;	// TODO: why no pixel center offset?
			p *= transform;
			p += origin;
			if (	p.x() >= 0 && p.x() < input.cols()-1
				&& p.y() >= 0 && p.y() < input.rows()-1)
				output(row, col) = interpolate(input, p);
		}
	}
	return output;
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
