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

// Stretch the rows of a matrix (this increases the amount of columns)
Eigen::MatrixXd stretch_rows(const Eigen::MatrixXd &input, const size_t cols)
{
	// Calculate stretch factor
	double factor = ((double) input.cols()) / cols;

	// Interpolate each row
	Eigen::MatrixXd output(input.rows(), cols);
	for (size_t row = 0; row < input.rows(); row++) {
		output(row, 0) = input(row, 0);				// HACK
		output(row, cols-1) = input(row, input.cols()-1);	// HACK
		for (size_t col = 1; col < cols-1; col++) {		// HACK
			double colsource = (col + 0.5) * factor - 0.5;
			double integral, fractional;
			fractional = std::modf(colsource, &integral);
			output(row, col) = (1-fractional)*input(row, (size_t) integral)
				+ fractional*input(row, (size_t) (integral+1));
		}
	}

	return output;
}

// Stretch the columns of a matrix (this increases the amount of rows)
Eigen::MatrixXd stretch_cols(const Eigen::MatrixXd &input, const size_t rows)
{
	// Calculate stretch factor
	double factor = ((double) input.rows()) / rows;

	// Interpolate each column
	Eigen::MatrixXd output(rows, input.cols());
	for (size_t col = 0; col < input.cols(); col++) {
		output(0, col) = input(0, col);				// HACK
		output(rows-1, col) = input(input.rows()-1, col);	// HACK
		for (size_t row = 1; row < rows-1; row++) {		// HACK
			double rowsource = (row + 0.5) * factor - 0.5;
			double integral, fractional;
			fractional = std::modf(rowsource, &integral);
			output(row, col) = (1-fractional)*input((size_t) integral, col)
				+ fractional*input((size_t) (integral+1), col);
		}
	}

	return output;
}

// Resize the matrix
// TODO: fix this up
Eigen::MatrixXd resize(const Eigen::MatrixXd &input, const size_t cols, const size_t rows)
{
	// x == col
	// y == row

	// Calculate stretch factor
	double colscale = (double)input.cols() / cols;
	double rowscale = (double)input.rows() / rows;

	// Interpolate other pixels
	Eigen::MatrixXd output(rows, cols);
	for (size_t col = 0; col < cols; col++) {
		double colsource = (col+0.5)*colscale - 0.5;
		double colint, colfract;
		colfract = std::modf(colsource, &colint);

		for (size_t row = 0; row < rows; row++) {
			double rowsource = (row+0.5)*rowscale - 0.5;
			double rowint, rowfract;
			rowfract = std::modf(rowsource, &rowint);

			std::cout << row << "x" << col << ": " << rowsource << "x" << colsource << std::endl;
			output(row, col) = 
				input((size_t) rowint, (size_t) colint)*(1-rowfract)*(1-colfract) +
				input((size_t) rowint+1, (size_t) colint)*(rowfract)*(1-colfract) +
				input((size_t) rowint, (size_t) colint+1)*(1-rowfract)*(colfract) +
				input((size_t) rowint+1, (size_t) colint+1)*(rowfract)*(colfract);
		}
	}

	return output;
}

double lerp(double c1, double c2, double v1, double v2, double x)
{
	if (v1==v2)
		return c1;
	double inc = ((c2-c1)/(v2 - v1)) * (x - v1);
	return c1 + inc;
}


// Write an ASCII PGM file
void pgmWrite(const Eigen::MatrixXd &data, std::string filename)
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
