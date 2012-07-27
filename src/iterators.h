//
// Configuration
//

// Include guard
#ifndef ITERATORS_H
#define ITERATORS_H

// System includes
#include <limits>

// OpenCV includes
#include <cv.h>

// Local includes
#include "auxiliary.h"


//
// Class definition
//
template <typename T>
class ImageIterator {
public:
	//
	// Construction and destruction
	//

	ImageIterator(const cv::Mat &i_image)
		: m_image(i_image)
	{
	}


	//
	// Basic I/O
	//

	virtual bool valid() const = 0;
	virtual T value() const = 0;


	//
	// Iteration interface
	//

	virtual bool hasNext() const = 0;
	virtual void next() = 0;
	virtual unsigned int samples() = 0;
	virtual void toFront() = 0;


	//
	// Lookup methods
	//

	// Conceptually this expands the list of indexes to a weighed one (in
	// which each index is repeated as many times as the pixel value it
	// represents), after which the median value of that array is located.
	void findWeighedMedian()
	{		
		double sum = 0;
		while (hasNext()) {
			sum += value();
			next();
		}
		toFront();

		double integral = 0;
		while (hasNext()) {
			integral += value();
			if (2*integral >= sum)
				break;
			next();
		}
	}

	// Look for the median of the weighed indexes, but take the square root
	// of the pixel values as weight
	void findWeighedSquaredMedian()
	{		
		double sum = 0;
		while (hasNext()) {
			sum += std::sqrt(value());
			next();
		}
		toFront();

		double integral = 0;
		while (hasNext()) {
			integral += std::sqrt(value());
			if (2*integral >= sum)
				break;
			next();
		}
	}

protected:
	const cv::Mat &image() const
	{
		return m_image;
	}
	T pixel(int y, int x) const
	{
		return m_image.at<T>(y, x);
	}

private:
	const cv::Mat &m_image;
};


//
// Column iterator
//

template <typename T>
class ColumnIterator : public ImageIterator<T> {
public:
	//
	// Construction and destruction
	//

	ColumnIterator(const cv::Mat &i_image, unsigned int i_column)
		: ImageIterator<T>(i_image), m_column(i_column)
	{
		// Check validity
		m_valid = true;
		if (i_column >= this->image().cols)
			m_valid = false;

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

	unsigned int row() const
	{
		return m_row;
	}
	
	T value() const
	{
		return value(m_row);
	}

	T value(unsigned int i_row) const
	{
		return this->pixel(i_row, m_column);
	}


	//
	// Iteration methods
	//

	bool hasNext() const
	{
		assert(m_valid);
		return m_row < this->image().rows;
	}

	void next()
	{
		// Advance
		m_row++;
	}

	unsigned int samples()
	{
		return this->image().rows;
	}

	void toFront()
	{
		m_row = 0;
	}

private:
	unsigned int m_column, m_row;
	bool m_valid;
};


//
// Line iterator
//

template <typename T>
class LineIterator : public ImageIterator<T> {
public:
	//
	// Construction and destruction
	//

	LineIterator(const cv::Mat &i_image, const Segment &i_segment)
		: ImageIterator<T>(i_image), m_segment(i_segment)
	{
		// Clip the segment against the image
		if (clip(
			Rectangle{
				Point{0, 0},
				Size{
					this->image().size().width,
					this->image().size().height
				}
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

	Point point() const
	{
		return m_p;
	}
	
	T value() const
	{
		return value(m_p);
	}

	T value(const Point &p) const
	{
		// Get fractional parts, floors and ceilings
		double x_fract, x_int;
		x_fract = std::modf(p.x, &x_int);
		double y_fract, y_int;
		y_fract = std::modf(p.y, &y_int);

		// Get the pixel value
		T pixel;
		assert(p.x >= 0 && p.y >= 0);	// 'cause *_fract end up with same sign
		if (x_fract < std::numeric_limits<double>::epsilon()
			&& y_fract < std::numeric_limits<double>::epsilon()) {
			pixel = this->pixel(y_int, x_int);
		} else {	// bilinear interpolation
			double upper_left, upper_right, bottom_left, bottom_right;
			double upper, bottom;

			bool x_pureint = x_fract < std::numeric_limits<double>::epsilon();
			bool y_pureint = y_fract < std::numeric_limits<double>::epsilon();

			// Calculate fractional coordinates
			upper_left = this->pixel(y_int, x_int);
			if (!x_pureint)
				upper_right = this->pixel(y_int, x_int+1);
			if (!y_pureint)
				bottom_left = this->pixel(y_int+1, x_int);
			if (!x_pureint && !y_pureint)
				bottom_right = this->pixel(y_int+1, x_int+1);

			// Calculate pixel value
			if (x_pureint) {
				pixel = (T) (upper_left*(1-y_fract) + bottom_left*y_fract);
			} else if (y_pureint) {
				pixel = (T) (upper_left*(1-x_fract) + upper_right*x_fract);
			} else {
				upper = upper_left*(1-x_fract) + upper_right*x_fract;
				bottom = bottom_left*(1-x_fract) + bottom_right*x_fract;
				
				pixel = (T) (upper*(1-y_fract) + bottom*y_fract);
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

	unsigned int samples()
	{
		return 1 + (unsigned int) std::floor(m_clipped.length());
	}

	void toFront()
	{
		m_p = m_clipped.begin;
		m_step = 0;
	}


	//
	// Transformations
	//

	LineIterator transformDomain(const Segment &i_segment) const
	{
		return LineIterator(this->image(), i_segment);
	}

private:
	const Segment m_segment;
	Segment m_clipped;
	bool m_valid;

	Point m_p, m_leap;
	unsigned int m_step;
};

#endif
