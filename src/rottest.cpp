//
// Configuration
//

// Standard library
#include <iostream>

// Local
#include "auxiliary.hpp"


//
// Main application
//

void printmat(const Eigen::MatrixXi input) {
    for (int row = 0; row < input.rows(); row++) {
        for (int col = 0; col < input.cols(); col++) {
            int pixel = input(row, col);
            int decimals = 1;
            if (pixel > 0)
                decimals = (int)std::log10(pixel) + 1;
            std::cout << pixel << std::string(4 - decimals, ' ');
        }
        std::cout << std::endl;
    }    
}

int main(int argc, char **argv) {
    // Manage arguments
    if (argc != 3) {
        std::cerr << "Please provide input and rotation angle" << std::endl;
        return 1;
    }
    std::string filename(argv[1]);
    int angle = std::stoi(argv[2]);

    // Read image
    Eigen::MatrixXi input_gray = readpgm(filename);
    printmat(input_gray);
    Eigen::MatrixXf input = gray2mat(input_gray);

    // Rotate image
    Point<float>::type origin((input.cols() - 1) / 2.0,
                              (input.rows() - 1) / 2.0);
    Eigen::MatrixXf input_rotated = rotate(input, origin, deg2rad(angle));

    // Output image
    std::cout << std::endl;
    Eigen::MatrixXi output = mat2gray(input_rotated);
    printmat(output);

    return 0;
}
