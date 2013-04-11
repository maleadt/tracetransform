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


//
// Structs
//

std::ostream& operator<<(std::ostream &stream, const Point& point) {
        stream << point.x() << "x" << point.y();
        return stream;
}


//
// Routines
//

Eigen::MatrixXd pgmRead(std::string filename)
{
        std::ifstream infile(filename);
        std::string inputLine = "";

        // First line: version
        getline(infile, inputLine);
        if (inputLine.compare("P2") != 0) {
                std::cerr << "readPGM: invalid PGM version " << inputLine << std::endl;
                // TODO: throw exception
        }

        // Second line: comment (optional)
        if (infile.peek() == '#')
                getline(infile, inputLine);

        // Continue with a stringstream
        std::stringstream ss;
        ss << infile.rdbuf();

        // Size
        size_t numrows = 0, numcols = 0;
        ss >> numcols >> numrows;
        Eigen::MatrixXd data(numrows, numcols);

        // Maxval
        size_t maxval;
        ss >> maxval;
        assert(maxval == 255);

        // Data
        double value;
        for (size_t row = 0; row < numrows; row++) {
                for (size_t col = 0; col < numcols; col++) {
                        ss >> value;
                        data(row, col) = value;
                }
        }
        infile.close();

        return data;
}

void pgmWrite(std::string filename, const Eigen::MatrixXd &data)
{
        std::ofstream outfile(filename);

        // First line: version
        outfile << "P2" << "\n";

        // Second line: size
        outfile << data.cols() << " " << data.rows() << "\n";

        // Third line: maxval
        outfile << 255 << "\n";

        // Data
        long pos = outfile.tellp();
        for (size_t row = 0; row < data.rows(); row++) {
                for (size_t col = 0; col < data.cols(); col++) {
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

void dataWrite(std::string filename, const Eigen::MatrixXd &data,
        const std::vector<std::string> &headers)
{
        assert(headers.size() == 0 || headers.size() == data.cols());

        // Calculate column width
        std::vector<size_t> widths(data.cols(), 0);
        for (size_t col = 0; col < data.cols(); col++) {
                if (headers.size() > 0)
                        widths[col] = headers[col].length();
                for (size_t row = 0; row < data.rows(); row++) {
                        double value = data(row, col);
                        size_t width = 3; // decimal, comma, 2 decimals
                        if (value > 1)
                                width += std::floor(std::log10(value));
                        if (value < 0)  // dash for negative numbers
                                width++;
                        if (width > widths[col])
                                widths[col] = width;
                }
                widths[col] += 2;       // add spacing
        }

        // Open file
        std::ofstream fd_data(filename);

        // Print headers
        if (headers.size() > 0) {
                fd_data << "%  ";
                fd_data << std::setiosflags(std::ios::fixed)
                        << std::setprecision(0);
                for (size_t col = 0; col < headers.size(); col++) {
                        fd_data << std::setw(widths[col]) << headers[col];
                }
                fd_data << "\n";
        }

        // Print data
        fd_data << std::setiosflags(std::ios::fixed) << std::setprecision(2);
        for (size_t row = 0; row < data.rows(); row++) {
                fd_data << "   ";
                for (size_t col = 0; col < data.cols(); col++) {
                        fd_data << std::setw(widths[col])
                                << data(row, col);
                }
                fd_data << "\n";
        }

        fd_data << std::flush;
        fd_data.close();
}

Eigen::MatrixXd gray2mat(const Eigen::MatrixXd &input)
{
        // Scale
        Eigen::MatrixXd output(input.rows(), input.cols());
        for (size_t col = 0; col < output.cols(); col++) {
                for (size_t row = 0; row < output.rows(); row++) {
                        output(row, col) = input(row, col) / 255.0;
                }
        }
        return output;
}

Eigen::MatrixXd mat2gray(const Eigen::MatrixXd &input)
{
        // Detect maximum
        double maximum = 0;
        for (size_t col = 0; col < input.cols(); col++) {
                for (size_t row = 0; row < input.rows(); row++) {
                        double pixel = input(row, col);
                        if (pixel > maximum)
                                maximum = pixel;
                }
        }

        // Scale
        Eigen::MatrixXd output(input.rows(), input.cols());
        for (size_t col = 0; col < output.cols(); col++) {
                for (size_t row = 0; row < output.rows(); row++) {
                        output(row, col) = input(row, col) * 255.0/maximum;
                }
        }
        return output;
}

double deg2rad(double degrees)
{
        return (degrees * M_PI / 180);
}

double interpolate(const Eigen::MatrixXd &source, const Point &p)
{
        assert(p.x() >= 0 && p.x() < source.cols()-1);
        assert(p.y() >= 0 && p.y() < source.rows()-1);

        // Get fractional and integral part of the coordinates
        double x_int, y_int;
        double x_fract = std::modf(p.x(), &x_int);
        double y_fract = std::modf(p.y(), &y_int);

        return    source((size_t)y_int, (size_t)x_int)*(1-x_fract)*(1-y_fract)
                + source((size_t)y_int, (size_t)x_int+1)*x_fract*(1-y_fract)
                + source((size_t)y_int+1, (size_t)x_int)*(1-x_fract)*y_fract
                + source((size_t)y_int+1, (size_t)x_int+1)*x_fract*y_fract;

}

Eigen::MatrixXd resize(const Eigen::MatrixXd &input, const size_t rows, const size_t cols)
{
        // Calculate transform matrix
        // TODO: use Eigen::Geometry
        Eigen::Matrix2d transform;
        transform <<    ((double) input.rows()) / rows, 0,
                        0, (((double) input.cols()) / cols);

        // Allocate output matrix
        Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows, cols);

        // Process all points
        // FIXME: borders are wrong (but this doesn't matter here since we
        //        only handle padded images)
        for (size_t col = 1; col < cols-1; col++) {
                for (size_t row = 1; row < rows-1; row++) {
                        Point p(col, row);
                        p += Eigen::RowVector2d(0.5, 0.5);
                        p *= transform;
                        p -= Eigen::RowVector2d(0.5, 0.5);
                        output(row, col) = interpolate(input, p);
                }
        }
        return output;
}

Eigen::MatrixXd rotate(const Eigen::MatrixXd &input, const Point &origin, const double angle)
{
        // Calculate transform matrix
        // TODO: use Eigen::Geometry
        // TODO: -angle?
        Eigen::Matrix2d transform;
        transform <<    std::cos(-angle), -std::sin(-angle),
                        std::sin(-angle),  std::cos(-angle);

        // Allocate output matrix
        Eigen::MatrixXd output = Eigen::MatrixXd::Zero(input.rows(), input.cols());

        // Process all points
        for (size_t col = 0; col < input.cols(); col++) {
                for (size_t row = 0; row < input.rows(); row++) {
                        Point p(col, row);
                        p -= origin;    // TODO: why no pixel center offset?
                        p *= transform;
                        p += origin;
                        if (    p.x() >= 0 && p.x() < input.cols()-1
                                && p.y() >= 0 && p.y() < input.rows()-1)
                                output(row, col) = interpolate(input, p);
                }
        }
        return output;
}

Eigen::MatrixXd pad(const Eigen::MatrixXd &image)
{
        // Pad the images so we can freely rotate without losing information
        Point origin(
                std::floor((image.cols() + 1) / 2.0) - 1,
                std::floor((image.rows() + 1) / 2.0) - 1);
        int rLast = (int) std::ceil(std::hypot(
                        image.cols() - 1 - origin.x() - 1,
                        image.rows() - 1 - origin.y() - 1)) + 1;
        int rFirst = -rLast;
        size_t nBins = (unsigned) (rLast - rFirst + 1);
        Eigen::MatrixXd image_padded = Eigen::MatrixXd::Zero(nBins, nBins);
        Point origin_padded(
                std::floor((image_padded.cols() + 1) / 2.0) - 1,
                std::floor((image_padded.rows() + 1) / 2.0) - 1);
        Point df = origin_padded - origin;
        for (size_t col = 0; col < image.cols(); col++) {
                for (size_t row = 0; row < image.rows(); row++) {
                        image_padded(row + (size_t) df.y(), col + (size_t) df.x())
                                = image(row, col);
                }
        }

        return image_padded;
}

double arithmetic_mean(const Eigen::VectorXd &input)
{
        if (input.size() == 0)
                return NAN;

        double sum = 0;
        for (size_t i = 0; i < input.size(); i++) {
                sum += input(i);
        }

        return sum / input.size();
}

double standard_deviation(const Eigen::VectorXd &input)
{
        if (input.size() <= 0)
                return NAN;

        double mean = arithmetic_mean(input);
        double sum = 0;
        for (size_t i = 0; i < input.size(); i++) {
                double diff = input(i) - mean;
                sum += diff*diff;
        }

        // NOTE: this is the default MATLAB interpretation, Wiki's std()
        //       uses vector.cols
        return std::sqrt(sum / (input.size()-1));
}

Eigen::VectorXd zscore(const Eigen::VectorXd &input)
{
        if (input.size() == 0)
                return Eigen::VectorXd();

        double mean = arithmetic_mean(input);
        double stdev = standard_deviation(input);

        Eigen::VectorXd transformed(input.size());
        for (size_t i = 0; i < input.size(); i++) {
                transformed(i) = (input(i) - mean) / stdev;
        }

        return transformed;
}
