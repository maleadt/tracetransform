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

// TODO: try to make these immutable

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

#endif
