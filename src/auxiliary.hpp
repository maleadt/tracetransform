//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_AUXILIARY_
#define _TRACETRANSFORM_AUXILIARY_

// Standard library
#include <iostream>
#include <string>
#include <vector>

// Eigen
#include <Eigen/Dense>


//
// Structs
//

typedef Eigen::RowVector2d Point;

std::ostream& operator<<(std::ostream &stream, const Point& point);


//
// Routines
//

// Read an ASCII PGM file
Eigen::MatrixXf pgmRead(std::string filename);

// Write an ASCII PGM file
void pgmWrite(std::string filename, const Eigen::MatrixXf &data);

// Write a MATLAB and gnuplot-compatible data file
void dataWrite(std::string filename, const Eigen::MatrixXf &data,
        const std::vector<std::string> &headers = std::vector<std::string>());

// Convert a grayscale image (range [0, 255]) to a matrix (range [0, 1]).
Eigen::MatrixXf gray2mat(const Eigen::MatrixXf &input);

// Convert a matrix (arbitrary values) to a grayscale image (range [0, 255]). This
// involves detecting the maximum value, and clamping that to 255.
Eigen::MatrixXf mat2gray(const Eigen::MatrixXf &input);

float deg2rad(float degrees);

float interpolate(const Eigen::MatrixXf &source, const Point &p);

Eigen::MatrixXf resize(const Eigen::MatrixXf &input, const size_t rows, const size_t cols);

Eigen::MatrixXf rotate(const Eigen::MatrixXf &input, const Point &origin, const float angle);

Eigen::MatrixXf pad(const Eigen::MatrixXf &image);

float arithmetic_mean(const Eigen::VectorXf &input);

float standard_deviation(const Eigen::VectorXf &input);

Eigen::VectorXf zscore(const Eigen::VectorXf &input);

template<typename T> int sgn(T val)
{
        return (T(0) < val) - (val < T(0));
}

std::string readable_si(double number, const std::string unit, double base = 1000);
std::string readable_size(double size);
std::string readable_frequency(double frequency);

#endif
