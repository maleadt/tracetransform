//
// Configuration
//

// Standard library
#include <iostream>
#include <vector>

// Local
#include "auxiliary.hpp"
#include "cudahelper/memory.hpp"
#include "kernels/rotate.hpp"


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
        std::cerr << "Please provide input and output filename" << std::endl;
        return 1;
    }
    std::string filename(argv[1]);
    int angle = std::stoi(argv[2]);

    // Read and upload image
    // Read image
    std::vector<Eigen::MatrixXi> image = readnetpbm(filename);
    assert(image.size() == 1);
    Eigen::MatrixXi input_gray = image[0];
    printmat(input_gray);
    Eigen::MatrixXf input_data = gray2mat(input_gray);
    CUDAHelper::GlobalMemory<float> *input =
        new CUDAHelper::GlobalMemory<float>(
            CUDAHelper::size_2d(input_data.rows(), input_data.cols()));
    input->upload(input_data.data());

    // Rotate and download image
    CUDAHelper::GlobalMemory<float> *input_rotated =
        new CUDAHelper::GlobalMemory<float>(input->sizes());
    rotate(input, input_rotated, -deg2rad(angle));
    delete input;
    Eigen::MatrixXf input_rotated_data(input_rotated->size(0),
                                       input_rotated->size(1));
    input_rotated->download(input_rotated_data.data());
    delete input_rotated;

    // Output image
    std::cout << std::endl;
    Eigen::MatrixXi output = mat2gray(input_rotated_data);
    printmat(output);

    return 0;
}
