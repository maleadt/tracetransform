//
// Configuration
//

// Standard library
#include <chrono>    // for microseconds, time_point, etc
#include <cstddef>   // for size_t
#include <exception> // for exception
#include <iostream>  // for operator<<, ostream, etc
#include <string>    // for operator+, string, etc
#include <vector>    // for vector

// Boost
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "transform.hpp"
#include "progress.hpp"


//
// Main application
//

enum class ProgramMode {
    CALCULATE,
    PROFILE,
    BENCHMARK
};

std::istream &operator>>(std::istream &in, ProgramMode &mode) {
    std::string name;
    in >> name;
    if (name == "calculate") {
        mode = ProgramMode::CALCULATE;
    } else if (name == "profile") {
        mode = ProgramMode::PROFILE;
    } else if (name == "benchmark") {
        mode = ProgramMode::BENCHMARK;
    } else {
        throw boost::program_options::validation_error(
            boost::program_options::validation_error::invalid_option_value);
    }
    return in;
}

int main(int argc, char **argv) {
    //
    // Initialization
    //

    // Program input
    ProgramMode mode;

    // List of functionals
    std::vector<TFunctionalWrapper> tfunctionals;
    std::vector<PFunctionalWrapper> pfunctionals;

    // List of inputs
    std::vector<std::string> inputs;

    // Declare named options
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h",
            "produce help message")
        ("quiet,q",
            "only display errors and warnings")
        ("verbose,v",
            "display some more details")
        ("debug,d",
            "display even more details")
        ("t-functional,T",
            boost::program_options::value<
                    std::vector<TFunctionalWrapper>>(&tfunctionals)
                ->required(),
            "T-functionals")
        ("p-functional,P",
            boost::program_options::value<
                    std::vector<PFunctionalWrapper>>(&pfunctionals),
            "P-functionals")
        ("angle,a",
            boost::program_options::value<unsigned int>()
                ->default_value(1),
            "angle stepsize")
        ("mode,m",
            boost::program_options::value<ProgramMode>(&mode)
                ->required(),
            "execution mode ('calculate', 'profile' or 'benchmark')")
        ("iterations,n",
            boost::program_options::value<unsigned int>(),
            "amount of iterations to run")
        ("inputs,i",
            boost::program_options::value<std::vector<std::string>>(&inputs)
                ->required(),
            "images to process")
    ;

    // Declare positional options
    boost::program_options::positional_options_description pod;
    pod.add("inputs", -1);

    // Parse the options
    boost::program_options::variables_map vm;
    try {
        store(boost::program_options::command_line_parser(argc, argv)
                  .options(desc)
                  .positional(pod)
                  .run(),
              vm);
    }
    catch (const std::exception &e) {
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
    }
    catch (const std::exception &e) {
        std::cerr << "Invalid usage: " << e.what() << std::endl;

        std::cout << desc << std::endl;
        return 1;
    }

    // Configure logging
    bool showProgress = false;
    if (vm.count("debug")) {
        logger.settings.threshold = trace;
        logger.settings.prefix_timestamp = true;
        logger.settings.prefix_level = true;
    } else if (vm.count("verbose"))
        logger.settings.threshold = debug;
    else if (vm.count("quiet"))
        logger.settings.threshold = warning;
    else
        showProgress = true;
    if (mode == ProgramMode::BENCHMARK)
        showProgress = false;

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
    else {
        clog(error) << "Cannot mix regular and orthonormal P-functionals" << std::endl;
        throw boost::program_options::validation_error(
            boost::program_options::validation_error::invalid_option_value,
            "pfunctionals");
    }


    //
    // Execution
    //

    Progress indicator(inputs.size());
    if (showProgress)
        indicator.start();
    for (const std::string &input : inputs) {
        // Get the image basename
        boost::filesystem::path path(input);
        if (!exists(path)) {
            clog(error) << "Input file does not exist" << std::endl;
            throw boost::program_options::validation_error(
                boost::program_options::validation_error::invalid_option_value,
                "inputs", input);
        }
        std::string basename = path.stem().string();

        // Load image components according to their type
        std::vector<Eigen::MatrixXi> components;
        if (boost::iequals(path.extension().string(), ".pgm") ||
            boost::iequals(path.extension().string(), ".ppm")) {
            components = readnetpbm(input);
        } else {
            clog(error) << "Unrecognized input file format" << std::endl;
            throw boost::program_options::validation_error(
                boost::program_options::validation_error::invalid_option_value,
                "inputs", input);
        }

        int i = 0;
        for (const auto &component : components) {
            // Generate a local basename
            i++;
            std::string component_name;
            if (components.size() == 1)
                component_name = basename;
            else
                component_name = basename + "_c" + std::to_string(i);

            // Preprocess the image
            Transformer transformer(gray2mat(component), component_name,
                                    vm["angle"].as<unsigned int>(),
                                    orthonormal);

            if (mode == ProgramMode::CALCULATE) {
                transformer.getTransform(tfunctionals, pfunctionals, true);
            } else if (mode == ProgramMode::PROFILE) {
                transformer.getTransform(tfunctionals, pfunctionals, false);
            } else if (mode == ProgramMode::BENCHMARK) {
                if (!vm.count("iterations"))
                    throw boost::program_options::required_option("iterations");

                // Allocate array for time measurements
                unsigned int iterations = vm["iterations"].as<unsigned int>();
                std::vector<std::chrono::time_point<
                    std::chrono::high_resolution_clock> > timings(iterations +
                                                                  1);

                // Warm-up
                transformer.getTransform(tfunctionals, pfunctionals, false);

                // Transform the image
                // NOTE: although the use of elapsed real time rather than CPU
                //       time might seem inaccurate, it is necessary because
                //       some of the ports execute code on non-CPU hardware
                std::chrono::time_point<std::chrono::high_resolution_clock>
                last, current;
                for (unsigned int n = 0; n < iterations; n++) {
                    last = std::chrono::high_resolution_clock::now();
                    transformer.getTransform(tfunctionals, pfunctionals, false);
                    current = std::chrono::high_resolution_clock::now();

                    clog(info) << "t_" << n + 1 << "="
                               << std::chrono::duration_cast<
                                      std::chrono::microseconds>(current - last)
                                          .count() /
                                      1000000.0 << std::endl;
                }
            }
        }

        if (showProgress)
            ++indicator;
    }

    return 0;
}
