//
// Configuration
//

// Standard library
#include <cstddef>
#include <exception>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>

// Boost
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Dense>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "transform.hpp"

//
// Main application
//

int main(int argc, char **argv)
{

        //
        // Initialization
        //

        // List of functionals
        std::vector<TFunctionalWrapper> tfunctionals;
        std::vector<PFunctionalWrapper> pfunctionals;

        // Declare named options
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()("help,h", "produce help message")("quiet,q",
                        "only display errors and warnings")("verbose,v",
                        "display some more details")("debug,d",
                        "display even more details")("input,i",
                        boost::program_options::value<std::string>()->required(),
                        "image to process")("t-functional,T",
                        boost::program_options::value<
                                        std::vector<TFunctionalWrapper>>(
                                        &tfunctionals)->required(),
                        "T-functionals")("p-functional,P",
                        boost::program_options::value<
                                        std::vector<PFunctionalWrapper>>(
                                        &pfunctionals), "P-functionals")(
                        "iterations,n",
                        boost::program_options::value<unsigned int>()->required(),
                        "amount of iterations to run");

        // Declare positional options
        boost::program_options::positional_options_description pod;
        pod.add("input", -1);

        // Parse the options
        boost::program_options::variables_map vm;
        try {
                store(
                                boost::program_options::command_line_parser(
                                                argc, argv).options(desc).positional(
                                                pod).run(), vm);
        } catch (const std::exception &e) {
                std::cerr << "Error " << e.what() << std::endl;

                std::cout << desc << std::endl;
                return 1;
        }

        // Display help
        if (vm.count("help")) {
                std::cout << desc << std::endl;
                return 0;
        }

        // Notify the user of errors
        try {
                notify(vm);
        } catch (const std::exception &e) {
                std::cerr << "Invalid usage: " << e.what() << std::endl;

                std::cout << desc << std::endl;
                return 1;
        }

        // Check for orthonormal P-functionals
        unsigned int orthonormal_count = 0;
        bool orthonormal;
        for (size_t p = 0; p < pfunctionals.size(); p++) {
                if (pfunctionals[p].functional == PFunctional::Hermite)
                        orthonormal_count++;
        }
        if (orthonormal_count == 0)
                orthonormal = false;
        else if (orthonormal_count == pfunctionals.size())
                orthonormal = true;
        else
                throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Cannot mix regular and orthonormal P-functionals");

        // Configure logging
        if (vm.count("debug")) {
                logger.settings.threshold = trace;
                logger.settings.prefix_timestamp = true;
                logger.settings.prefix_level = true;
        } else if (vm.count("verbose"))
                logger.settings.threshold = debug;
        else if (vm.count("quiet"))
                logger.settings.threshold = warning;

        //
        // Image processing
        //

        // Read the image
        Eigen::MatrixXf input = gray2mat(
                        readpgm(vm["input"].as<std::string>()));

        // Allocate array for time measurements
        unsigned int iterations = vm["iterations"].as<unsigned int>();
        std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>>
                timings(iterations + 1);

        // Warm-up
        Transformer transformer(input, orthonormal);
        transformer.getTransform(tfunctionals, pfunctionals, false);

        // Transform the image
        timings[0] = std::chrono::system_clock::now();
        for (unsigned int n = 0; n < iterations; n++) {
                transformer.getTransform(tfunctionals, pfunctionals, false);
                timings[n + 1] = std::chrono::system_clock::now();
        }

        // Get iteration durations
        std::vector<long int> durations(iterations);
        for (unsigned int n = 0; n < iterations; n++) {
                durations[n] = std::chrono::duration_cast<std::chrono::milliseconds>
                        (timings[n + 1] - timings[n]).count();
                clog(debug) << "Iteration " << n << ": " << durations[n] << " ms." << std::endl;
        }

        // Calculate some statistics
        double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
        double mean = sum / durations.size();
        double sq_sum = std::inner_product(durations.begin(), durations.end(),
                        durations.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / durations.size() - mean * mean);

        clog(info) << "Total execution time for " << iterations
                        << " iterations: " << sum << " ms." << std::endl;
        clog(info) << "Average execution time: " << mean << " +/- " << stdev
                        << " ms." << std::endl;

        return 0;
}
