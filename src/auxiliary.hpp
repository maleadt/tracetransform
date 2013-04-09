//
// Configuration
//

// Include guard
#ifndef TRACETRANSFORM_AUXILIARY_HPP
#define TRACETRANSFORM_AUXILIARY_HPP

// Standard library
#include <iostream>
#include <string>
#include <vector>

// Eigen
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::size_t
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
Eigen::MatrixXd pgmRead(std::string filename);

// Write an ASCII PGM file
void pgmWrite(std::string filename, const Eigen::MatrixXd &data);

// Write a MATLAB and gnuplot-compatible data file
void dataWrite(std::string filename, const Eigen::MatrixXd &data,
        const std::vector<std::string> &headers = std::vector<std::string>());

// Convert a grayscale image (range [0, 255]) to a matrix (range [0, 1]).
Eigen::MatrixXd gray2mat(const Eigen::MatrixXd &input);

// Convert a matrix (arbitrary values) to a grayscale image (range [0, 255]). This
// involves detecting the maximum value, and clamping that to 255.
Eigen::MatrixXd mat2gray(const Eigen::MatrixXd &input);

double deg2rad(double degrees);

double interpolate(const Eigen::MatrixXd &source, const Point &p);

Eigen::MatrixXd resize(const Eigen::MatrixXd &input, const unsigned int rows, const unsigned int cols);

Eigen::MatrixXd rotate(const Eigen::MatrixXd &input, const Point &origin, const double angle);

double arithmetic_mean(const Eigen::VectorXd &input);

double standard_deviation(const Eigen::VectorXd &input);

Eigen::VectorXd zscore(const Eigen::VectorXd &input);

#endif
