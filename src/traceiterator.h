//
// Configuration
//

// Include guard
#ifndef TRACEITERATOR_H
#define TRACEITERATOR_H

// System includes
#include <limits>

// OpenCV includes
#include <cv.h>

// Local includes
#include "auxiliary.h"


//
// Class definition
//

// This class allows to trace a line within a matrix, without having to rotate
// it completely. It uses bilinear interpolation to get values of pixels not
// addressable using integer indexes
class TraceIterator {
public:
	//
	// Construction and destruction
	//

	TraceIterator(const cv::Mat &i_image, const Segment &i_segment)
		: m_image(i_image), m_segment(i_segment)
	{
		// Clip the segment against the image
		if (clip(
			Rectangle{
				Point{0, 0},
				Size{m_image.size().width, m_image.size().height}
			},
			m_segment,
			m_clipped
		)){
			// The segment is valid, calculate the leap distance
			m_valid = true;
			m_leap = Point{m_clipped.rcx(),	m_clipped.rcy()};
		} else {
			// No part of the image falls within the image, so bail
			// out as soon as possible
			m_valid = false;
		}

		// Initialize trace
		toFront();
	}


	//
	// Basic I/O
	//

	bool valid() const
	{
		return m_valid;
	}

	const Segment &segment() const
	{
		return m_segment;
	}

	const Point &point()
	{
		return m_p;
	}

	uchar value() const
	{
		return value(m_p);
	}
	
	uchar value(const Point &p) const
	{
		// Get fractional parts, floors and ceilings
		double x_fract, x_int;
		x_fract = std::modf(p.x, &x_int);
		double y_fract, y_int;
		y_fract = std::modf(p.y, &y_int);

		// Get the pixel value
		uchar pixel;
		assert(p.x >= 0 && p.y >= 0);	// 'cause *_fract end up with same sign
		if (x_fract < std::numeric_limits<double>::epsilon()
			&& y_fract < std::numeric_limits<double>::epsilon()) {
			pixel = m_image.at<uchar>((int)y_int, (int)x_int);
		} else {	// bilinear interpolation
			double upper_left, upper_right, bottom_left, bottom_right;
			double upper, bottom;

			bool x_pureint = x_fract < std::numeric_limits<double>::epsilon();
			bool y_pureint = y_fract < std::numeric_limits<double>::epsilon();

			// Calculate fractional coordinates
			upper_left = m_image.at<uchar>((int)y_int, (int)x_int);
			if (!x_pureint)
				upper_right = m_image.at<uchar>((int)y_int, (int)x_int+1);
			if (!y_pureint)
				bottom_left = m_image.at<uchar>((int)y_int+1, (int)x_int);
			if (!x_pureint && !y_pureint)
				bottom_right = m_image.at<uchar>((int)y_int+1, (int)x_int+1);

			// Calculate pixel value
			if (x_pureint) {
				pixel = (uchar) (upper_left*(1-y_fract) + bottom_left*y_fract);
			} else if (y_pureint) {
				pixel = (uchar) (upper_left*(1-x_fract) + upper_right*x_fract);
			} else {
				upper = upper_left*(1-x_fract) + upper_right*x_fract;
				bottom = bottom_left*(1-x_fract) + bottom_right*x_fract;
				
				pixel = (uchar) (upper*(1-y_fract) + bottom*y_fract);
			}
		}

		return pixel;
	}


	//
	// Iteration methods
	//

	bool hasNext() const
	{
		assert(m_valid);
		return m_step <= m_clipped.length();
	}

	void next()
	{
		// Advance
		m_step++;
		m_p = m_clipped.begin + m_leap*m_step;		

		// Clamp any invalid pixels. This can happen due to rounding
		// errors: given a long enough trace, the multiplication of
		// m_steps with the leap vector can induce small errors, causing
		// an out of bounds lookup in getPixel()
		//
		// FIXME: this is very expensive to be put in the hotpath
		double x_low = std::min(m_clipped.begin.x, m_clipped.end.x);
		double x_high = std::max(m_clipped.begin.x, m_clipped.end.x);
		double y_low = std::min(m_clipped.begin.y, m_clipped.end.y);
		double y_high = std::max(m_clipped.begin.y, m_clipped.end.y);
		if (m_p.x < x_low || m_p.x > x_high
			|| m_p.y < y_low || m_p.y > y_high) {
			m_p = m_clipped.end;
			assert(m_step+1 > m_clipped.length());
		}
	}

	void toFront()
	{
		m_p = m_clipped.begin;
		m_step = 0;
	}


	//
	// Transformations
	//

	TraceIterator transformDomain(const Segment &i_segment) const
	{
		return TraceIterator(m_image, i_segment);
	}

private:
	const cv::Mat &m_image;
	const Segment m_segment;
	Segment m_clipped;
	bool m_valid;

	Point m_p, m_leap;
	unsigned int m_step;
};

// Functional signatures
typedef std::function<double(TraceIterator&)> Functional;

#endif
