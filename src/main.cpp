//
// Configuration
//

// System includes
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <sys/stat.h>

// Boost
#include <boost/program_options.hpp>
#include <boost/optional.hpp>
#include <boost/format.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Local includes
#include "auxiliary.h"
extern "C" {
        #include "functionals.h"
}
#include "wrapper.h"
#include "tracetransform.h"
#include "circusfunction.h"

// Algorithm parameters
#define ANGLE_INTERVAL          1
#define DISTANCE_INTERVAL       1


//
// Program option parsers
//

struct TFunctional {
        std::string name;
        FunctionalWrapper *wrapper;
};

std::istream& operator>>(std::istream& in, TFunctional& tfunctional)
{
    std::string token;
    in >> token;
    if (token == "0") {
        tfunctional.name = "T0";
        tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctionalRadon);
        } else if (token == "1") {
                tfunctional.name = "T1";
                tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional1);
        } else if (token == "2") {
                tfunctional.name = "T2";
                tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional2);
        } else if (token == "3") {
                tfunctional.name = "T3";
                tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional3);
        } else if (token == "4") {
                tfunctional.name = "T4";
                tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional4);
        } else if (token == "5") {
                tfunctional.name = "T5";
                tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional5);
        } else {
                throw boost::program_options::validation_error(
                        boost::program_options::validation_error::invalid_option_value,
                        "Unknown T-functional");
        }
    return in;
}

struct PFunctional
{
        enum
        {
                REGULAR,
                HERMITE
        } type;

        std::string name;
        FunctionalWrapper *wrapper;

        boost::optional<unsigned int> order;
};

std::istream& operator>>(std::istream& in, PFunctional& pfunctional)
{
        std::string token;
        in >> token;
        if (token == "1") {
                pfunctional.name = "P1";
                pfunctional.wrapper = new SimpleFunctionalWrapper(PFunctional1);
                pfunctional.type = PFunctional::REGULAR;
        } else if (token == "2") {
                pfunctional.name = "P2";
                pfunctional.wrapper = new SimpleFunctionalWrapper(PFunctional2);
                pfunctional.type = PFunctional::REGULAR;
        } else if (token == "3") {
                pfunctional.name = "P3";
                pfunctional.wrapper = new SimpleFunctionalWrapper(PFunctional3);
                pfunctional.type = PFunctional::REGULAR;
        } else if (token[0] == 'H') {
                unsigned int order;
                if (token.size() < 2)
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Missing order parameter for Hermite P-functional");
                try {
                        order = boost::lexical_cast<unsigned int>(token.substr(1));
                }
                catch(boost::bad_lexical_cast &) {
                        throw boost::program_options::validation_error(
                                boost::program_options::validation_error::invalid_option_value,
                                "Unparseable order parameter for Hermite P-functional");
                }

                pfunctional.name = boost::str(boost::format("H%d") % order);
                pfunctional.wrapper = new GenericFunctionalWrapper<unsigned int, unsigned int>(PFunctionalHermite);
                pfunctional.order = order;
                pfunctional.type = PFunctional::HERMITE;
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
        std::vector<TFunctional> tfunctionals;
        std::vector<PFunctional> pfunctionals;

        // Declare named options
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
                ("help",
                "       produce help message")
                ("verbose",
                        "display some more details")
                ("input,i",
                        boost::program_options::value<std::string>()
                        ->required(),
                        "image to process")
                ("output,o",
                        boost::program_options::value<std::string>()
                        ->default_value("circus.dat"),
                        "where to write the output circus data")
                ("t-functional,T",
                        boost::program_options::value<std::vector<TFunctional>>(&tfunctionals)
                        ->required(),
                        "T-functionals")
                ("p-functional,P",
                        boost::program_options::value<std::vector<PFunctional>>(&pfunctionals),
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

        // Check for orthonormal P-functionals
        size_t orthonormal_count = 0;
        for (size_t p = 0; p < pfunctionals.size(); p++) {
                if (pfunctionals[p].type == PFunctional::HERMITE)
                        orthonormal_count++;
        }
        bool orthonormal;
        if (orthonormal_count == 0)
                orthonormal = false;
        else if (orthonormal_count == pfunctionals.size())
                orthonormal = true;
        else
                throw boost::program_options::validation_error(
                        boost::program_options::validation_error::invalid_option_value,
                        "Cannot mix regular and orthonormal P-functionals"); 
        

        //
        // Image processing
        //
        
        // Read the image
        Eigen::MatrixXd input = pgmRead(vm["input"].as<std::string>());
        input = gray2mat(input);

        // Orthonormal P-functionals need a stretched image in order to ensure a
        // square sinogram
        if (orthonormal) {
                int ndiag = (int) std::ceil(360.0/ANGLE_INTERVAL);
                int size = (int) std::ceil(ndiag/std::sqrt(2));
                input = resize(input, size, size);
        }

        // Pad the image so we can freely rotate without losing information
        Point origin(
                std::floor((input.cols() - 1) / 2.0),
                std::floor((input.rows() - 1) / 2.0));
        int diagonal = (int) std::ceil(std::hypot(
                input.cols(),           // 2 pixel border since we use
                input.rows())) + 2;     // bilinear interpolation
        Eigen::MatrixXd input_padded = Eigen::MatrixXd::Zero(diagonal, diagonal);
        Point origin_padded(
                std::floor((input_padded.cols() - 1) / 2.0),
                std::floor((input_padded.rows() - 1) / 2.0));
        Point df = origin_padded - origin;
        for (int col = 0; col < input.cols(); col++) {
                for (int row = 0; row < input.rows(); row++) {
                        input_padded(row + (int) df.y(), col + (int) df.x())
                                = input(row, col);
                }
        }

        // Allocate a matrix for all output data to reside in
        Eigen::MatrixXd output(
                360 / ANGLE_INTERVAL,
                tfunctionals.size() * pfunctionals.size());

        // Process all T-functionals
        if (vm.count("verbose"))
                std::cerr << "Calculating";
        for (size_t t = 0; t < tfunctionals.size(); t++) {
                // Calculate the trace transform sinogram
                if (vm.count("verbose"))
                        std::cerr << " " << tfunctionals[t].name << "..." << std::flush;
                Eigen::MatrixXd sinogram = getTraceTransform(
                        input_padded,
                        ANGLE_INTERVAL,
                        DISTANCE_INTERVAL,
                        tfunctionals[t].wrapper
                );

#ifndef NDEBUG
                // Save the sinogram image
                std::stringstream fn_trace_image;
                fn_trace_image << "trace_" << tfunctionals[t].name << ".pgm";
                pgmWrite(fn_trace_image.str(), mat2gray(sinogram));

                // Save the sinogram data
                std::stringstream fn_trace_data;
                fn_trace_data << "trace_" << tfunctionals[t].name << ".dat";
                dataWrite(fn_trace_data.str(), sinogram);
#endif

                // Orthonormal functionals require the nearest orthonormal sinogram
                unsigned int sinogram_center;
                if (orthonormal)
                        sinogram = nearest_orthonormal_sinogram(sinogram, sinogram_center);

                // Process all P-functionals
                for (size_t p = 0; p < pfunctionals.size(); p++) {
                        // Configure any extra parameters
                        if (pfunctionals[p].type == PFunctional::HERMITE)
                                dynamic_cast<GenericFunctionalWrapper<unsigned int, unsigned int>*>
                                        (pfunctionals[p].wrapper)
                                        ->configure(*pfunctionals[p].order, sinogram_center);

                        // Calculate the circus function
                        if (vm.count("verbose"))
                                std::cerr << " " << pfunctionals[p].name << "..." << std::flush;
                        Eigen::VectorXd circus = getCircusFunction(
                                sinogram,
                                pfunctionals[p].wrapper
                        );

                        // Normalize
                        Eigen::VectorXd normalized = zscore(circus);

                        // Copy the data
                        assert(normalized.size() == output.rows());
                        output.col(t*pfunctionals.size() + p) = normalized;
                }
        }
        if (vm.count("verbose"))
                std::cerr << "\n";

        // Save the output data
        if (pfunctionals.size() > 0) {          
                std::vector<std::string> headers;
                for (int tp = 0; tp < output.cols(); tp++) {
                        size_t t = tp / (unsigned) pfunctionals.size();
                        size_t p = tp % (unsigned) pfunctionals.size();
                        std::stringstream header;
                        header << tfunctionals[t].name << "-"
                                << pfunctionals[p].name;
                        headers.push_back(header.str());
                }
                dataWrite(vm["output"].as<std::string>(), output, headers);
        }

        return 0;
}
