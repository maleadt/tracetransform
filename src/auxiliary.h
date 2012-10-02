//
// Configuration
//

// Include guard
#ifndef AUXILIARY_H
#define AUXILIARY_H

// OpenCV includes
#include <cv.h>


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

// Convert a grayscale image (range [0, 255]) to a matrix (range [0, 1]).
cv::Mat gray2mat(const cv::Mat &grayscale)
{
	cv::Mat matrix(grayscale.size(), CV_64FC1);
	grayscale.convertTo(matrix, CV_64FC1, 1/255., 0);
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
