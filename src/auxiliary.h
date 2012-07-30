//
// Configuration
//

// Include guard
#ifndef AUXILIARY_H
#define AUXILIARY_H

// System includes
#include <complex>

// OpenCV includes
#include <cv.h>


//
// Data
//

// Point clipping bitcodes
typedef int ClipCode;
const int INSIDE = 0;	// 0000
const int LEFT = 1;	// 0001
const int RIGHT = 2;	// 0010
const int BOTTOM = 4;	// 0100
const int TOP = 8;	// 1000


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

struct Segment
{
	double dx() const {
		return end.x - begin.x;
	}
	double dy() const {
		return end.y - begin.y;
	}

	double length() const {
		return std::hypot(dx(), dy());
	}

	double rcx() const {
		return dx() / length();
	}
	double rcy() const {
		return dy() / length();
	}

	Point begin;
	Point end;
};

std::ostream& operator<<(std::ostream &stream, const Segment& segment) {
	stream << segment.begin << " → " << segment.end;
	return stream;
}

struct Size
{
	int width;
	int height;
};

struct Rectangle
{
	double xmin() const
	{
		return begin.x;
	}

	double ymin() const
	{
		return begin.y;
	}

	double xmax() const
	{
		return begin.x + size.width-1;
	}

	double ymax() const
	{
		return begin.x + size.height-1;
	}

	Point begin;
	Size size;
};



//
// Routines
//

inline double deg2rad(double degrees)
{
  return (degrees * M_PI / 180.0);
}

inline double rad2deg(double degrees)
{
	return (degrees * 180.0 / M_PI);
}

// Clip a point against a rectangle
ClipCode clip_point(const Rectangle &rectangle, const Point &point)
{
	// Initialize as being inside of clip window
	ClipCode code = INSIDE;

	if (point.x < rectangle.xmin())		// to the left of clip window
		code |= LEFT;
	else if (point.x > rectangle.xmax())	// to the right of clip window
		code |= RIGHT;
	if (point.y < rectangle.ymin())		// below the clip window
		code |= BOTTOM;
	else if (point.y > rectangle.ymax())	// above the clip window
		code |= TOP;

	return code;
}

// Cohen–Sutherland line clipping algorithm
bool clip(const Rectangle &rectangle, const Segment &segment, Segment &clipped)
{
	// Clip endpoints against rectangle
	ClipCode code0 = clip_point(rectangle, segment.begin);
	ClipCode code1 = clip_point(rectangle, segment.end);
	bool accept = false;

	clipped = segment;
	while (true) {
		if (!(code0 | code1)) {		// Trivially accept
			accept = true;
			break;
		} else if (code0 & code1) {	// Trivially reject
			break;
		} else {			// Calculate the line segment to
						// clip from an outside point to
						// an intersection with the clip
						// edge
			Point p;

			// At least one endpoint is outside the clip rectangle; pick it.
			ClipCode codeOut = code0? code0 : code1;

			// Now find the intersection point
			if (codeOut & TOP) {			// above the clip rectangle
				p.x = clipped.begin.x
					+ (clipped.end.x - clipped.begin.x)
					* (rectangle.ymax() - clipped.begin.y)
					/ (clipped.end.y - clipped.begin.y);
				p.y = rectangle.ymax();
			} else if (codeOut & BOTTOM) {	// below the clip rectangle
				p.x = clipped.begin.x
					+ (clipped.end.x - clipped.begin.x)
					* (rectangle.ymin() - clipped.begin.y)
					/ (clipped.end.y - clipped.begin.y);
				p.y = rectangle.ymin();
			} else if (codeOut & RIGHT) {	// to the right of clip rectangle
				p.y = clipped.begin.y
					+ (clipped.end.y - clipped.begin.y)
					* (rectangle.xmax() - clipped.begin.x)
					/ (clipped.end.x - clipped.begin.x);
				p.x = rectangle.xmax();
			} else /* if (codeOut & LEFT) */ {	// to the left of clip rectangle
				p.y = clipped.begin.y
					+ (clipped.end.y - clipped.begin.y)
					* (rectangle.xmin() - clipped.begin.x)
					/ (clipped.end.x - clipped.begin.x);
				p.x = rectangle.xmin();
			}

			// NOTE: last check is a plain else, because without it
			// the algorithm can fall into an infinite loop in case
			// a line crosses more than two segments

			// Now we move outside point to intersection point to
			// clip and get ready for next pass.
			if (codeOut == code0) {
				clipped.begin = p;
				code0 = clip_point(rectangle, clipped.begin);
			} else {
				clipped.end = p;
				code1 = clip_point(rectangle, clipped.end);
			}
		}
	}
	return accept;
}

// Convert a grayscale image (range [0, 255]) to a matrix (range [0, 1]).
cv::Mat gray2mat(const cv::Mat &grayscale)
{
	cv::Mat matrix(grayscale.size(), CV_64FC1);
	grayscale.convertTo(matrix, CV_64FC1, 1/255., 0);
	return matrix;
}

// Convert a matrix (arbitrary values) to a grayscale image (range [0, 255]). This
// involves detecting the maximum value, and clamping that to 255.
cv::Mat mat2gray(const cv::Mat &matrix)
{
	double maximum = 0;
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			double pixel = matrix.at<double>(i, j);
			if (pixel > maximum)
				maximum = pixel;
		}
	}
	cv::Mat grayscale(matrix.size(), CV_8UC1);
	matrix.convertTo(grayscale, CV_8UC1, 255.0/maximum, 0);
	return grayscale;
}

template <typename T>
std::vector<std::complex<double>> dft(const std::vector<std::complex<T>> &sample)
{
	std::vector<std::complex<double>> output(sample.size());
	for(size_t i = 0; i < sample.size(); i++) 
	{
		output[i] = std::complex<double>(0, 0);
		double arg = -1.0 * 2.0 * M_PI * (double)i / (double)sample.size();
		for(size_t j = 0; j < sample.size(); j++) 
		{
			double cosarg = std::cos(j * arg);
			double sinarg = std::sin(j * arg);
			output[i] += std::complex<double>(
				((double)sample[j].real() * cosarg - (double)sample[j].imag() * sinarg),
				((double)sample[j].real() * sinarg + (double)sample[j].imag() * cosarg)
			);
		}
	}
	return output;
}

double hermite_polynomial(unsigned int order, double x) {
	switch (order) {
		case 0:
			return 1.0;

		case 1:
			return 2.0*x;

		default:
			return 2.0*x*hermite_polynomial(order-1, x)
				-2.0*(order-1)*hermite_polynomial(order-2, x);
	}
}

unsigned int factorial(unsigned int n)
{
	return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

double hermite_function(unsigned int order, double x) {
	return hermite_polynomial(order, x) / (
			std::sqrt(std::pow(2, order) * factorial(order) * std::sqrt(M_PI))
			* std::exp(x*x/2)
		);
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
