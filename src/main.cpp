//
// Configuration
//

// System includes
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <fstream>
#include <sys/stat.h>
#include <boost/program_options.hpp>

// Library includes
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Local includes
#include "auxiliary.h"
extern "C" {
	#include "functionals.h"
}
#include "tracetransform.h"
#include "circusfunction.h"

// Algorithm parameters
#define ANGLE_INTERVAL		1
#define DISTANCE_INTERVAL	1

// Namespaces
namespace po = boost::program_options;


//
// Auxiliary
//

struct Profiler
{
	Profiler()
	{
		t1 = clock();
	}

	void stop()
	{
		t2 = clock();		
	}

	double elapsed()
	{
		return (double)(t2-t1)/CLOCKS_PER_SEC;
	}

	time_t t1, t2;
};


//
// Main
//

enum PFunctionalType {
       REGULAR,
       HERMITE
};

int main(int argc, char **argv)
{
	// Declare named options
	po::options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("debug", "write debug images and data while calculating")
	    ("verbose", "display some more details")
	    ("profile", "profile the different steps of the algorithm")
	    ("image,I", po::value<std::string>(), "image to process")
	    ("t-functional,T", po::value<std::vector<unsigned int>>(), "T-functionals")
	    ("p-functional,P", po::value<std::vector<unsigned int>>(), "P-functionals")
	    ("h-functional,H", po::value<std::vector<unsigned int>>(), "Hermite P-functionals")
	;

	// Declare positional options
	po::positional_options_description p;
	p.add("image", -1);

	po::variables_map vm;
	po::store(po::command_line_parser(argc, argv).
	          options(desc).positional(p).run(), vm);
	po::notify(vm);  

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	// Get the chosen T-functionals
	std::vector<Functional> tfunctionals;
	std::vector<void*> tfunctional_arguments;
	std::vector<std::string> tfunctional_names;
	if (vm.count("t-functional") == 0) {
		std::cerr << "Error: specify at least 1 T-functional" << std::endl;
		std::cout << desc << std::endl;
		return 0;
	}
	for (unsigned int functional : vm["t-functional"].as<std::vector<unsigned int>>()) {
		std::stringstream name;
		switch (functional) {
		case 0:
			tfunctionals.push_back(TFunctionalRadon);
			tfunctional_arguments.push_back(nullptr);
			break;
		case 1:
			tfunctionals.push_back(TFunctional1);
			tfunctional_arguments.push_back(nullptr);
			break;
		case 2:
			tfunctionals.push_back(TFunctional2);
			tfunctional_arguments.push_back(nullptr);
			break;
		case 3:
			tfunctionals.push_back(TFunctional3);
			tfunctional_arguments.push_back(nullptr);
			break;
		case 4:
			tfunctionals.push_back(TFunctional4);
			tfunctional_arguments.push_back(nullptr);
			break;
		case 5:
			tfunctionals.push_back(TFunctional5);
			tfunctional_arguments.push_back(nullptr);
			break;
		default:
			std::cerr << "Error: invalid T-functional provided" << std::endl;
			return 1;
		}
		name << "T" << functional;
		tfunctional_names.push_back(name.str());
	}

	// Get the chosen P-functional
	std::vector<Functional> pfunctionals;
	std::vector<void*> pfunctional_arguments;
	std::vector<std::string> pfunctional_names;
	if (vm.count("p-functional") > 0 && vm.count("h-functional") > 0) {
		std::cerr << "Error: cannot mix orthonormal and regular P-functionals" << std::endl;
		return 1;
	}
	if (vm.count("p-functional") > 0) {
		for (unsigned int functional : vm["p-functional"].as<std::vector<unsigned int>>()) {
			std::stringstream name;
			switch (functional) {
			case 1:
				pfunctionals.push_back(PFunctional1);
				pfunctional_arguments.push_back(nullptr);
				break;
			case 2:
				pfunctionals.push_back(PFunctional2);
				pfunctional_arguments.push_back(nullptr);
				break;
			case 3:
				pfunctionals.push_back(PFunctional3);
				pfunctional_arguments.push_back(nullptr);
				break;
			default:
				std::cerr << "Error: invalid P-functional provided" << std::endl;
				return 1;
			}
			name << "P" << functional;
			pfunctional_names.push_back(name.str());
		}
	}
	if (vm.count("h-functional") > 0) {
		for (unsigned int functional : vm["h-functional"].as<std::vector<unsigned int>>()) {
			std::stringstream name;
			pfunctionals.push_back(PFunctionalHermite);
			pfunctional_arguments.push_back(new ArgumentsHermite{functional, 0});
			name << "H" << functional;
			pfunctional_names.push_back(name.str());
		}
	}

	// Read the image
	if (vm.count("image") == 0) {
		std::cerr << "Error: no image specified" << std::endl;
		std::cout << desc << std::endl;
		return 0;
	}
	Eigen::MatrixXd input = pgmRead(vm["image"].as<std::string>());
	input = gray2mat(input);

	// Orthonormal P-functionals need a stretched image in order to ensure
	// a square sinogram
	if (vm.count("h-functional") > 0) {
		int ndiag = (int) std::ceil(360.0/ANGLE_INTERVAL);
		int size = (int) std::ceil(ndiag/std::sqrt(2));
		input = resize(input, size, size);
	}

	// Pad the image so we can freely rotate without losing information
	Point origin(std::floor((input.cols()-1)/2.0), std::floor((input.rows()-1)/2.0));
	int rLast = std::ceil(std::hypot(input.cols() - 1 - origin.x(), input.rows() - 1 - origin.y()));
	int rFirst = -rLast;
	int nBins = rLast - rFirst + 1;
	Eigen::MatrixXd input_padded = Eigen::MatrixXd::Zero(nBins, nBins);
	Point origin_padded(std::floor((input_padded.cols() - 1)/2.0), std::floor((input_padded.rows()-1)/2.0));
	Point df = origin_padded - origin;
	for (int col = 0; col < input.cols(); col++) {
		for (int row = 0; row < input.rows(); row++) {
			input_padded(row + df.y(), col + df.x()) = input(row, col);
		}
	}

	// Save profiling data
	std::vector<double> tfunctional_runtimes(tfunctionals.size());
	std::vector<double> pfunctional_runtimes(pfunctionals.size(), 0);

	// Allocate a matrix for all output data to reside in
	Eigen::MatrixXd output (360 / ANGLE_INTERVAL, tfunctionals.size()*pfunctionals.size());

	// Process all T-functionals
	Profiler mainprofiler;
	if (vm.count("verbose"))
		std::cerr << "Calculating";
	for (size_t t = 0; t < tfunctionals.size(); t++) {
		// Calculate the trace transform sinogram
		if (vm.count("verbose"))
			std::cerr << " " << tfunctional_names[t] << "..." << std::flush;
		Profiler tprofiler;
		Eigen::MatrixXd sinogram = getTraceTransform(
			input_padded,
			ANGLE_INTERVAL,		// angle resolution
			DISTANCE_INTERVAL,	// distance resolution
			tfunctionals[t],
			tfunctional_arguments[t]
		);
		tprofiler.stop();
		tfunctional_runtimes[t] = tprofiler.elapsed();

		if (vm.count("debug") || pfunctionals.size() == 0) {
			// Save the sinogram image
			std::stringstream fn_trace_image;
			fn_trace_image << "trace_" << tfunctional_names[t] << ".pgm";
			pgmWrite(fn_trace_image.str(), mat2gray(sinogram));

			// Save the sinogram data
			std::stringstream fn_trace_data;
			fn_trace_data << "trace_" << tfunctional_names[t] << ".dat";
			dataWrite(fn_trace_data.str(), sinogram);
		}

		// Hermite functionals require the nearest orthonormal sinogram
		unsigned int sinogram_center;
		if (vm.count("h-functional") > 0)
			sinogram = nearest_orthonormal_sinogram(sinogram, sinogram_center);

		// Process all P-functionals
		for (size_t p = 0; p < pfunctionals.size(); p++) {
			// Extra parameters to functional
			if (vm.count("h-functional") > 0)
				((ArgumentsHermite*)pfunctional_arguments[p])->center = sinogram_center;

			// Calculate the circus function
			if (vm.count("verbose"))
				std::cerr << " " << pfunctional_names[p] << "..." << std::flush;
			Profiler pprofiler;
			Eigen::VectorXd circus = getCircusFunction(
				sinogram,
				pfunctionals[p],
				pfunctional_arguments[p]
			);
			pprofiler.stop();
			pfunctional_runtimes[p] += pprofiler.elapsed();

			// Normalize
			Eigen::VectorXd normalized = zscore(circus);

			// Copy the data
			assert(normalized.size() == output.rows());
			output.col(t*pfunctionals.size() + p) = normalized;
		}
	}
	if (vm.count("verbose"))
		std::cerr << "\n";
	mainprofiler.stop();

	// Print runtime measurements	
	if (vm.count("profile")) {
		std::cerr << "t(total) = " << mainprofiler.elapsed()
			<< " s" << std::endl;
		for (size_t t = 0; t < tfunctionals.size(); t++) {
			std::cerr << "t(" << tfunctional_names[t] << ") = "
				<< tfunctional_runtimes[t] << " s\n";
		}
		for (size_t p = 0; p < pfunctionals.size(); p++) {
			std::cerr << "t(" << pfunctional_names[p] << ") = "
				<< pfunctional_runtimes[p] / tfunctionals.size() << " s\n";
		}
	}

	// Save the output data
	if (pfunctionals.size() > 0) {		
		std::vector<std::string> headers;
		for (size_t tp = 0; tp < output.cols(); tp++) {
			size_t t = tp / pfunctionals.size();
			size_t p = tp % pfunctionals.size();
			std::stringstream header;
			header << tfunctional_names[t] << "-"
				<< pfunctional_names[p];
			headers.push_back(header.str());
		}
		dataWrite("circus.dat", output, headers);
	}

	return 0;
}
