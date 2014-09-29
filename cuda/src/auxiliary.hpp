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

// Local
#include "global.hpp"


//
// Routines
//

// Read an ASCII PPM file
std::vector<Eigen::MatrixXi> readnetpbm(std::string filename);

// Write an ASCII PGM file
void writepgm(std::string filename, const Eigen::MatrixXi &data);

// Write a CSV file
void writecsv(std::string filename, const Eigen::MatrixXf &data);

// Convert a grayscale image (range [0, 255]) to a matrix (range [0, 1]).
Eigen::MatrixXf gray2mat(const Eigen::MatrixXi &input);

// Convert a matrix (arbitrary values) to a grayscale image (range [0, 255]).
// This
// involves detecting the maximum value, and clamping that to 255.
Eigen::MatrixXi mat2gray(const Eigen::MatrixXf &input);

float deg2rad(float degrees);

float interpolate(const Eigen::MatrixXf &source, const Point<float>::type &p);

Eigen::MatrixXf resize(const Eigen::MatrixXf &input, const size_t rows,
                       const size_t cols);

Eigen::MatrixXf pad(const Eigen::MatrixXf &image);

template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

std::string readable_si(double number, const std::string unit,
                        double base = 1000);
std::string readable_size(double size);
std::string readable_frequency(double frequency);

#endif
