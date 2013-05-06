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

// Boost
#include <boost/program_options.hpp>
#include <boost/format.hpp>

// Eigen
#include <Eigen/Dense>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
#include "transform.hpp"


//
// Program option parsers
//

std::istream& operator>>(std::istream& in, TFunctionalWrapper& wrapper)
{
        in >> wrapper.name;
        wrapper.name = "T" + wrapper.name;
        if (wrapper.name == "T0") {
                wrapper.functional = TFunctional::Radon;
        } else if (wrapper.name == "T1") {
                wrapper.functional = TFunctional::T1;
        } else if (wrapper.name == "T2") {
                wrapper.functional = TFunctional::T2;
        } else if (wrapper.name == "T3") {
                wrapper.functional = TFunctional::T3;
        } else if (wrapper.name == "T4") {
                wrapper.functional = TFunctional::T4;
        } else if (wrapper.name == "T5") {
                wrapper.functional = TFunctional::T5;
        } else {
                throw boost::program_options::validation_error(
                        boost::program_options::validation_error::invalid_option_value,
                        "Unknown T-functional");
        }
        return in;
}

std::istream& operator>>(std::istream& in, PFunctionalWrapper& wrapper)
{
        in >> wrapper.name;
        if (isdigit(wrapper.name[0]))
            wrapper.name = "P" + wrapper.name;
        if (wrapper.name == "P1") {
                wrapper.functional = PFunctional::P1;
        } else if (wrapper.name == "P2") {
                wrapper.functional = PFunctional::P2;
        } else if (wrapper.name == "P3") {
                wrapper.functional = PFunctional::P3;
        } else if (wrapper.name[0] == 'H') {
                wrapper.functional = PFunctional::Hermite;
                if (wrapper.name.size() < 2)
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Missing order parameter for Hermite P-functional");
                try {
                        wrapper.arguments.order = boost::lexical_cast<unsigned int>(wrapper.name.substr(1));
                }
                catch(boost::bad_lexical_cast &) {
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Unparseable order parameter for Hermite P-functional");
                }
        } else {
                throw boost::program_options::validation_error(
                        boost::program_options::validation_error::invalid_option_value,
                        "Unknown P-functional");
        }
    return in;
}


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
        desc.add_options()
                ("help,h",
                        "produce help message")
                ("quiet,q",
                        "only display errors and warnings")
                ("verbose,v",
                        "display some more details")
                ("debug,d",
                        "display even more details")
                ("input,i",
                        boost::program_options::value<std::string>()
                        ->required(),
                        "image to process")
                ("t-functional,T",
                        boost::program_options::value<std::vector<TFunctionalWrapper>>(&tfunctionals)
                        ->required(),
                        "T-functionals")
                ("p-functional,P",
                        boost::program_options::value<std::vector<PFunctionalWrapper>>(&pfunctionals),
                        "P-functionals")
        ;

        // Declare positional options
        boost::program_options::positional_options_description pod;
        pod.add("input", -1);

        // Parse the options
        boost::program_options::variables_map vm;
        try {
                store(boost::program_options::command_line_parser(argc, argv)
                        .options(desc).positional(pod).run(), vm);
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
        Eigen::MatrixXf input = gray2mat(readpgm(vm["input"].as<std::string>()));

        // Transform the image
        Transformer transformer(input);
        transformer.getTransform(tfunctionals, pfunctionals);

        clog(debug) << "Exiting" << std::endl;
        return 0;
}
