//
// Configuration
//

// Header include
#include "auxiliary.hpp"

// Standard library
#include <fstream>
#include <iomanip>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <stdexcept>


//
// Routines
//

Eigen::MatrixXi readpgm(std::string filename) {
    std::ifstream infile(filename);
    if (!infile.good())
        throw std::runtime_error("could not open input file");
    std::string inputLine = "";

    // First line: version
    getline(infile, inputLine);
    if (inputLine.compare("P2") != 0)
        throw std::runtime_error("invalid PGM version");

    // Second line: comment (optional)
    if (infile.peek() == '#')
        getline(infile, inputLine);

    // Continue with a stringstream
    std::stringstream ss;
    ss << infile.rdbuf();

    // Size
    size_t numrows = 0, numcols = 0;
    ss >> numcols >> numrows;
    Eigen::MatrixXi data(numrows, numcols);

    // Maxval
    size_t maxval;
    ss >> maxval;
    assert(maxval == 255);

    // Data
    float value;
    for (size_t row = 0; row < numrows; row++) {
        for (size_t col = 0; col < numcols; col++) {
            ss >> value;
            data(row, col) = value;
        }
    }
    infile.close();

    return data;
}

void writepgm(std::string filename, const Eigen::MatrixXi &data) {
    std::ofstream outfile(filename);

    // First line: version
    outfile << "P2"
            << "\n";

    // Second line: size
    outfile << data.cols() << " " << data.rows() << "\n";

    // Third line: maxval
    outfile << 255 << "\n";

    // Data
    long pos = outfile.tellp();
    for (int row = 0; row < data.rows(); row++) {
        for (int col = 0; col < data.cols(); col++) {
            outfile << data(row, col);
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

void writecsv(std::string filename, const Eigen::MatrixXf &data) {
    // Open file
    std::ofstream fd_data(filename);

    // Print data
    for (int row = 0; row < data.rows(); row++) {
        for (int col = 0; col < data.cols(); col++) {
            fd_data << data(row, col);
            if (col < data.cols() - 1)
                fd_data << ", ";
        }
        fd_data << "\n";
    }

    fd_data << std::flush;
    fd_data.close();
}

Eigen::MatrixXf gray2mat(const Eigen::MatrixXi &input) {
    // Scale
    Eigen::MatrixXf output(input.rows(), input.cols());
    for (int col = 0; col < output.cols(); col++) {
        for (int row = 0; row < output.rows(); row++) {
            output(row, col) = input(row, col) / 255.0;
        }
    }
    return output;
}

Eigen::MatrixXi mat2gray(const Eigen::MatrixXf &input) {
    // Detect maximum
    float maximum = 0;
    for (int col = 0; col < input.cols(); col++) {
        for (int row = 0; row < input.rows(); row++) {
            float pixel = input(row, col);
            if (pixel > maximum)
                maximum = pixel;
        }
    }

    // Scale
    Eigen::MatrixXi output(input.rows(), input.cols());
    for (int col = 0; col < output.cols(); col++) {
        for (int row = 0; row < output.rows(); row++) {
            output(row, col) = input(row, col) * 255.0 / maximum;
        }
    }
    return output;
}

float deg2rad(float degrees) { return (degrees * M_PI / 180); }

float interpolate(const Eigen::MatrixXf &source, const Point<float>::type &p) {
    assert(p.x() >= 0 && p.x() < source.cols() - 1);
    assert(p.y() >= 0 && p.y() < source.rows() - 1);

    // Get fractional and integral part of the coordinates
    float x_int, y_int;
    float x_fract = std::modf(p.x(), &x_int);
    float y_fract = std::modf(p.y(), &y_int);

    return source(y_int, x_int) * (1 - x_fract) * (1 - y_fract) +
           source(y_int, x_int + 1) * x_fract * (1 - y_fract) +
           source(y_int + 1, x_int) * (1 - x_fract) * y_fract +
           source(y_int + 1, x_int + 1) * x_fract * y_fract;
}

Eigen::MatrixXf resize(const Eigen::MatrixXf &input, const size_t rows,
                       const size_t cols) {
    // Calculate transform matrix
    // TODO: use Eigen::Geometry
    Eigen::Matrix2f transform;
    transform << ((float)input.rows()) / rows, 0, 0,
        (((float)input.cols()) / cols);

    // Allocate output matrix
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(rows, cols);

    // Process all points
    // FIXME: borders are wrong (but this doesn't matter here since we
    //        only handle padded images)
    for (size_t col = 1; col < cols - 1; col++) {
        for (size_t row = 1; row < rows - 1; row++) {
            Point<float>::type p(col, row);
            p += Eigen::RowVector2f(0.5, 0.5);
            p *= transform;
            p -= Eigen::RowVector2f(0.5, 0.5);
            output(row, col) = interpolate(input, p);
        }
    }
    return output;
}

Eigen::MatrixXf rotate(const Eigen::MatrixXf &input,
                       const Point<float>::type &origin, const float angle) {
    // Calculate transform matrix
    // TODO: use Eigen::Geometry
    Eigen::Matrix2f transform;
    transform << std::cos(angle), -std::sin(angle), std::sin(angle),
        std::cos(angle);

    // Allocate output matrix
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(input.rows(), input.cols());

    // Process all points
    for (int col = 0; col < input.cols(); col++) {
        for (int row = 0; row < input.rows(); row++) {
            Point<float>::type p(col, row);
            p -= origin; // TODO: why no pixel center offset?
            p *= transform;
            p += origin;
            if (p.x() >= 0 && p.x() < input.cols() - 1 && p.y() >= 0 &&
                p.y() < input.rows() - 1)
                output(row, col) = interpolate(input, p);
        }
    }
    return output;
}

Eigen::MatrixXf pad(const Eigen::MatrixXf &image) {
    // Pad the images so we can freely rotate without losing information
    Point<float>::type origin(std::floor((image.cols() + 1) / 2.0) - 1,
                              std::floor((image.rows() + 1) / 2.0) - 1);
    int rLast = (int)std::ceil(std::hypot(image.cols() - 1 - origin.x() - 1,
                                          image.rows() - 1 - origin.y() - 1)) +
                1;
    int rFirst = -rLast;
    int nBins = (rLast - rFirst + 1);
    Eigen::MatrixXf image_padded = Eigen::MatrixXf::Zero(nBins, nBins);
    Point<float>::type origin_padded(
        std::floor((image_padded.cols() + 1) / 2.0) - 1,
        std::floor((image_padded.rows() + 1) / 2.0) - 1);
    Point<float>::type df = origin_padded - origin;
    for (int col = 0; col < image.cols(); col++) {
        for (int row = 0; row < image.rows(); row++) {
            image_padded(row + (int)df.y(), col + (int)df.x()) =
                image(row, col);
        }
    }

    return image_padded;
}

float arithmetic_mean(const Eigen::VectorXf &input) {
    if (input.size() == 0)
        return NAN;

    float sum = 0;
    for (int i = 0; i < input.size(); i++) {
        sum += input(i);
    }

    return sum / input.size();
}

float standard_deviation(const Eigen::VectorXf &input) {
    if (input.size() <= 0)
        return NAN;

    float mean = arithmetic_mean(input);
    float sum = 0;
    for (int i = 0; i < input.size(); i++) {
        float diff = input(i) - mean;
        sum += diff * diff;
    }

    // NOTE: this is the default MATLAB interpretation, Wiki's std()
    //       uses vector.cols
    return std::sqrt(sum / (input.size() - 1));
}

Eigen::VectorXf zscore(const Eigen::VectorXf &input) {
    if (input.size() == 0)
        return Eigen::VectorXf();

    float mean = arithmetic_mean(input);
    float stdev = standard_deviation(input);

    Eigen::VectorXf transformed(input.size());
    for (int i = 0; i < input.size(); i++) {
        transformed(i) = (input(i) - mean) / stdev;
    }

    return transformed;
}

std::string readable_si(double number, const std::string unit, double base) {
    const std::vector<std::string> dimensions = { "",  "K", "M", "G", "T",
                                                  "P", "E", "Z", "Y" };

    int i = 0;
    while (number >= base) {
        number /= base;
        i++;
    }

    std::stringstream ss;
    ss << std::setiosflags(std::ios::fixed) << std::setprecision(2) << number
       << " " << dimensions[i] << unit;
    return ss.str();
}

std::string readable_size(double size) { return readable_si(size, "iB", 1024); }

std::string readable_frequency(double frequency) {
    return readable_si(frequency, "Hz");
}
