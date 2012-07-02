//
// Configuration
//

// Include guard
#ifndef AUXILIARY_H
#define AUXILIARY_H

// System includes
#include <cmath>

// OpenCV includes
#include <cv.h>


//
// Data
//

const unsigned int palette_jet[] = {
	0x000090,
	0x000fff,
	0x0090ff,
	0x0fffee,
	0x90ff70,
	0xffee00,
	0xff7000,
	0xee0000,
	0x7f0000
};

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
	Point() {
	}
	Point(double i_x, double i_y) : x(i_x), y(i_y) {
	}

	Point operator*(const double factor) const {
		return Point(x*factor, y*factor);
	}

	Point operator+(const Point& term) const {
		return Point(x+term.x, y+term.y);
	}

	Point operator-(const Point& term) const {
		return Point(x-term.x, y-term.y);
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
	Segment() {
	}
	Segment(Point i_begin, Point i_end) : begin(i_begin), end(i_end) {
	}

	double dx() const {
		return end.x - begin.x;
	}
	double dy() const {
		return end.y - begin.y;
	}

	double length() const {
		return std::sqrt(dx()*dx() + dy()*dy());
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

struct Rectangle
{
	Rectangle() {
	}
	Rectangle(double i_xmin, double i_ymin, double i_xmax, double i_ymax)
		: xmin(i_xmin), ymin(i_ymin), xmax(i_xmax), ymax(i_ymax) {
	}

	double xmin, ymin;
	double xmax, ymax;
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

cv::Mat equalize(const cv::Mat &image)
{
	assert(image.type() == CV_8UC1);
	cv::Mat equalized(image.size(), image.type());

	// Determine the range
	uchar min = 255, max = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			uchar pixel = image.at<uchar>(i, j);
			if (min > pixel)
				min = pixel;
			if (pixel > max)
				max = pixel;
		}
	}
	if (min == 0 && max == 255)
		return image;

	// Equalize the range
	uchar range = (uchar) (max - min);
	double adaption = 255.0 / range;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			uchar pixel = image.at<uchar>(i, j);
			equalized.at<uchar>(i, j) = (uchar) ((pixel - min) * adaption);
		}
	}
	return equalized;
}

cv::Mat colormap(const cv::Mat &image)
{
	assert(image.type() == CV_8UC1);
	cv::Mat colormap(image.size(), image.type());
	cvtColor(colormap, colormap, CV_GRAY2BGR);

	// Equalize the range
	cv::Mat equalized = equalize(image);

	// Partition the colors
	double partitioning = 255.0 / (sizeof(palette_jet)/sizeof(unsigned int) - 1);
	for (int i = 0; i < equalized.rows; i++) {
		for (int j = 0; j < equalized.cols; j++) {
			uchar pixel = equalized.at<uchar>(i, j);

			double c = pixel / partitioning;
			double c_fract, c_int;
			c_fract = std::modf(c, &c_int);

			uchar r, g, b;
			if (c_fract < std::numeric_limits<double>::epsilon()) {
				r = (palette_jet[(int)c] >> 16) & 0xFF;
				g = (palette_jet[(int)c] >> 8) & 0xFF;
				b = palette_jet[(int)c] & 0xFF;
			} else {
				uchar r1, g1, b1;
				r1 = (palette_jet[(int)c_int] >> 16) & 0xFF;
				g1 = (palette_jet[(int)c_int] >> 8) & 0xFF;
				b1 = palette_jet[(int)c_int] & 0xFF;

				uchar r2, g2, b2;
				r2 = (palette_jet[(int)c_int+1] >> 16) & 0xFF;
				g2 = (palette_jet[(int)c_int+1] >> 8) & 0xFF;
				b2 = palette_jet[(int)c_int+1] & 0xFF;

				// Linear interpolation
				r = (uchar) (r1*(1-c_fract) + r2*c_fract);
				g = (uchar) (g1*(1-c_fract) + g2*c_fract);
				b = (uchar) (b1*(1-c_fract) + b2*c_fract);
			}

			colormap.at<cv::Vec3b>(i, j)[0] = b;
			colormap.at<cv::Vec3b>(i, j)[1] = g;
			colormap.at<cv::Vec3b>(i, j)[2] = r;
		}
	}

	return colormap;
}

// Clip a point against a rectangle
ClipCode clip_point(const Rectangle &rectangle, const Point &point)
{
	// Initialize as being inside of clip window
	ClipCode code = INSIDE;

	if (point.x < rectangle.xmin)		// to the left of clip window
		code |= LEFT;
	else if (point.x > rectangle.xmax)	// to the right of clip window
		code |= RIGHT;
	if (point.y < rectangle.ymin)		// below the clip window
		code |= BOTTOM;
	else if (point.y > rectangle.ymax)	// above the clip window
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
					* (rectangle.ymax - clipped.begin.y)
					/ (clipped.end.y - clipped.begin.y);
				p.y = rectangle.ymax;
			} else if (codeOut & BOTTOM) {	// below the clip rectangle
				p.x = clipped.begin.x
					+ (clipped.end.x - clipped.begin.x)
					* (rectangle.ymin - clipped.begin.y)
					/ (clipped.end.y - clipped.begin.y);
				p.y = rectangle.ymin;
			} else if (codeOut & RIGHT) {	// to the right of clip rectangle
				p.y = clipped.begin.y
					+ (clipped.end.y - clipped.begin.y)
					* (rectangle.xmax - clipped.begin.x)
					/ (clipped.end.x - clipped.begin.x);
				p.x = rectangle.xmax;
			} else /* if (codeOut & LEFT) */ {	// to the left of clip rectangle
				p.y = clipped.begin.y
					+ (clipped.end.y - clipped.begin.y)
					* (rectangle.xmin - clipped.begin.x)
					/ (clipped.end.x - clipped.begin.x);
				p.x = rectangle.xmin;
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

#endif
