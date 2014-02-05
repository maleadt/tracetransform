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
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

// Eigen
#include <Eigen/Dense>

// CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// OpenMP
#include <omp.h>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "transform.hpp"
#include "progress.hpp"
#include "cudahelper/errorhandling.hpp"


//
// Main application
//

// TODO: make this an enum class when ICC 14 is released
enum ProgramMode {
    CALCULATE,
    PROFILE,
    BENCHMARK
};

std::istream &operator>>(std::istream &in, ProgramMode &mode) {
    std::string name;
    in >> name;
    if (name == "calculate") {
        mode = CALCULATE;
    } else if (name == "profile") {
        mode = PROFILE;
    } else if (name == "benchmark") {
        mode = BENCHMARK;
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
            "image to process")
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
    if (mode == BENCHMARK)
        showProgress = false;

#ifdef WITH_CULA
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
#endif


    //
    // Execution
    //

    // Check for CUDA devices
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    clog(debug) << "Found " << num_gpus << " CUDA device(s)." << std::endl;
    if (num_gpus < 1) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    } else {
        for (int i = 0; i < num_gpus; i++) {
            cudaDeviceProp prop;
            CUDAHelper::checkError(cudaGetDeviceProperties(&prop, i));

            clog(trace) << " --- General Information for device " << i
                        << " --- " << std::endl;
            clog(trace) << "     Name: " << prop.name << std::endl;
            clog(trace) << "     Compute capability: " << prop.major << "."
                        << prop.minor << std::endl;
            clog(trace) << "     Clock rate: "
                        << readable_frequency(prop.clockRate * 1024)
                        << std::endl;
            clog(trace) << "     Device copy overlap: "
                        << (prop.deviceOverlap ? "enabled" : "disabled")
                        << std::endl;
            clog(trace) << "     Kernel execution timeout: "
                        << (prop.kernelExecTimeoutEnabled ? "enabled"
                                                          : "disabled")
                        << std::endl;

            clog(trace) << " --- Memory Information for device " << i << " --- "
                        << std::endl;
            clog(trace) << "     Total global memory: "
                        << readable_size(prop.totalGlobalMem) << std::endl;
            clog(trace) << "     Total constant memory: "
                        << readable_size(prop.totalConstMem) << std::endl;
            clog(trace) << "     Total memory pitch: "
                        << readable_size(prop.memPitch) << std::endl;
            clog(trace) << "     Texture alignment: " << prop.textureAlignment
                        << std::endl;

            clog(trace) << " --- Multiprocessing Information for device " << i
                        << " --- " << std::endl;
            clog(trace) << "     Multiprocessor count: "
                        << prop.multiProcessorCount << std::endl;
            clog(trace) << "     Shared memory per processor: "
                        << readable_size(prop.sharedMemPerBlock) << std::endl;
            clog(trace) << "     Registers per processor: " << prop.regsPerBlock
                        << std::endl;
            clog(trace) << "     Threads in warp: " << prop.warpSize
                        << std::endl;
            clog(trace) << "     Maximum threads per block: "
                        << prop.maxThreadsPerBlock << std::endl;
            clog(trace) << "     Maximum thread dimensions: ("
                        << prop.maxThreadsDim[0] << ", "
                        << prop.maxThreadsDim[1] << ", "
                        << prop.maxThreadsDim[2] << ")" << std::endl;
            clog(trace) << "     Maximum grid dimensions: ("
                        << prop.maxGridSize[0] << ", " << prop.maxGridSize[1]
                        << ", " << prop.maxGridSize[2] << ")" << std::endl;
        }
    }
    CUDAHelper::checkError(cudaDeviceSynchronize());

    Progress indicator(inputs.size());
    if (showProgress)
        indicator.start();
    omp_set_num_threads(num_gpus);
    #pragma omp parallel for
    for (auto it = inputs.begin(); it < inputs.end(); it++) {
        const std::string &input = *it;

        // Select a GPU
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        CUDAHelper::checkError(cudaSetDevice(cpu_thread_id % num_gpus));
        int gpu_id = -1;
        CUDAHelper::checkError(cudaGetDevice(&gpu_id));
        #pragma omp critical
                clog(debug) << "Processing '" << input << "' on GPU " << gpu_id
                            << " in CPU thread " << cpu_thread_id << " (of "
                            << num_cpu_threads << ")" << std::endl;

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
#ifdef WITH_CULA
            Transformer transformer(gray2mat(component), component_name,
                                    vm["angle"].as<unsigned int>(),
                                    orthonormal);
#else
            Transformer transformer(gray2mat(component), component_name,
                                    vm["angle"].as<unsigned int>());
#endif

            if (mode == CALCULATE) {
                transformer.getTransform(tfunctionals, pfunctionals, true);
            } else if (mode == PROFILE) {
                cudaProfilerStart();
                transformer.getTransform(tfunctionals, pfunctionals, false);
                cudaProfilerStop();
            } else if (mode == BENCHMARK) {
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
                // time might seem inaccurate, it is necessary because
                // some of the ports execute code on non-CPU hardware
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

        #pragma omp critical
        if (showProgress)
            ++indicator;
    }

    CUDAHelper::checkError(cudaDeviceReset());

    return 0;
}
