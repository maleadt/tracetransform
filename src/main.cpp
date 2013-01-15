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
#define ANGLE_INTERVAL		1
#define DISTANCE_INTERVAL	1


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

struct TFunctional {
	std::string name;
	FunctionalWrapper *wrapper;
};

typedef std::vector<TFunctional> TFunctionalList;

void validate(boost::any &v, const std::vector<std::string> &values,
              TFunctionalList*, int)
{
	using namespace boost::program_options;
	TFunctional tfunctional;

    // Parse functional selector.
    if (values.size() == 0)
        throw boost::program_options::validation_error(validation_error::invalid_option_value, "No T-functional specified");
    std::string functional = values.at(0);
    if (functional == "0") {
    	tfunctional.name = "T0";
    	tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctionalRadon);
	} else if (functional == "1") {
    	tfunctional.name = "T1";
    	tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional1);
	} else if (functional == "2") {
    	tfunctional.name = "T2";
    	tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional2);
	} else if (functional == "3") {
    	tfunctional.name = "T3";
    	tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional3);
	} else if (functional == "4") {
    	tfunctional.name = "T4";
    	tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional4);
	} else if (functional == "5") {
    	tfunctional.name = "T5";
    	tfunctional.wrapper = new SimpleFunctionalWrapper(TFunctional5);
	} else {
		throw boost::program_options::validation_error(validation_error::invalid_option_value, "Unknown T-functional");
	}

	// Manage list of options
	// TODO: let Boost do this?
	if (!v.empty()) {
		boost::any_cast<TFunctionalList>(v).push_back(tfunctional);
	} else {
		v = boost::any(TFunctionalList({tfunctional}));
	}
}

struct PFunctional {
	std::string name;
	FunctionalWrapper *wrapper;

	bool orthonormal;
};

typedef std::vector<PFunctional> PFunctionalList;

void validate(boost::any &v, const std::vector<std::string> &values,
              PFunctionalList*, int)
{
	using namespace boost::program_options;
	PFunctional pfunctional;

    // Parse functional selector.
    if (values.size() == 0)
        throw boost::program_options::validation_error(validation_error::invalid_option_value, "No P-functional specified");
	std::string functional = values.at(0);
	if (functional == "1") {
    	pfunctional.name = "P1";
    	pfunctional.wrapper = new SimpleFunctionalWrapper(PFunctional1);
    	pfunctional.orthonormal = false;
	} else if (functional == "2") {
    	pfunctional.name = "P2";
    	pfunctional.wrapper = new SimpleFunctionalWrapper(PFunctional2);
    	pfunctional.orthonormal = false;
	} else if (functional == "3") {
    	pfunctional.name = "P3";
    	pfunctional.wrapper = new SimpleFunctionalWrapper(PFunctional3);
    	pfunctional.orthonormal = false;
	} else if (functional == "H") {
		unsigned int order;
		if (values.size() < 2)
	        throw boost::program_options::validation_error(
	        	validation_error::invalid_option_value,
	        	"Missing order parameter for Hermite P-functional");
    	try {
        	order = boost::lexical_cast<unsigned int>(values.at(1));
	    }
	    catch(boost::bad_lexical_cast &) {
	        throw boost::program_options::validation_error(
	        	validation_error::invalid_option_value,
	        	"Unparseable order parameter for Hermite P-functional");
	    }

		pfunctional.name = boost::str(boost::format("H%d") % order);
		pfunctional.wrapper = new HermiteFunctionalWrapper(PFunctionalHermite, order);
    	pfunctional.orthonormal = true;
	} else {
    		throw boost::program_options::validation_error(validation_error::invalid_option_value, "Unknown P-functional");
	}

	// Manage list of options
	// TODO: let Boost do this?
	if (!v.empty()) {
		boost::any_cast<PFunctionalList>(v).push_back(pfunctional);
	} else {
		v = boost::any(PFunctionalList({pfunctional}));
	}
}

int main(int argc, char **argv)
{
	//
	// Initialization
	//
	
	using namespace boost::program_options;
	
	// List of functionals
	TFunctionalList tfunctionals;
	PFunctionalList pfunctionals;

	// Declare named options
	options_description desc("Allowed options");
	desc.add_options()
	    ("help", "produce help message")
	    ("debug", "write debug images and data while calculating")
	    ("verbose", "display some more details")
	    ("profile", "profile the different steps of the algorithm")
	    ("image,I", value<std::string>(), "image to process")
	    ("t-functional,T", value<TFunctionalList>(&tfunctionals)->multitoken(), "T-functionals")
	    ("p-functional,P", value<PFunctionalList>(&pfunctionals)->multitoken(), "P-functionals")
	;

	// Declare positional options
	positional_options_description pod;
	pod.add("image", -1);

	variables_map vm;
	store(command_line_parser(argc, argv).
	          options(desc).positional(pod).run(), vm);
	notify(vm);  

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	// Check for orthonormal P-functionals
	size_t orthonormal_count = 0;
	for (size_t p = 0; p < pfunctionals.size(); p++) {
		if (pfunctionals[p].orthonormal)
			orthonormal_count++;
	}
	bool orthonormal;
	if (orthonormal_count == 0)
		orthonormal = false;
	else if (orthonormal_count == pfunctionals.size())
		orthonormal = true;
	else
		throw boost::program_options::validation_error(
			validation_error::invalid_option_value,
			"Cannot mix regular and orthonormal P-functionals"); 
	

	//
	// Main
	//
	
	// Read the image
	if (vm.count("image") == 0) {
		// TODO: use Boost::po::->set_required
		std::cerr << "Error: no image specified" << std::endl;
		std::cout << desc << std::endl;
		return 0;
	}
	Eigen::MatrixXd input = pgmRead(vm["image"].as<std::string>());
	input = gray2mat(input);

	// Orthonormal P-functionals need a stretched image in order to ensure
	// a square sinogram
	if (orthonormal) {
		int ndiag = (int) std::ceil(360.0/ANGLE_INTERVAL);
		int size = (int) std::ceil(ndiag/std::sqrt(2));
		input = resize(input, size, size);
	}

	// Pad the image so we can freely rotate without losing information
	Point origin(std::floor((input.cols()-1)/2.0), std::floor((input.rows()-1)/2.0));
	int rLast = (int) std::ceil(std::hypot(
		input.cols() - 1- origin.x(),
		input.rows() - 1 - origin.y()));
	int rFirst = -rLast;
	int nBins = rLast - rFirst + 1;
	Eigen::MatrixXd input_padded = Eigen::MatrixXd::Zero(nBins, nBins);
	Point origin_padded(std::floor((input_padded.cols() - 1)/2.0), std::floor((input_padded.rows()-1)/2.0));
	Point df = origin_padded - origin;
	for (int col = 0; col < input.cols(); col++) {
		for (int row = 0; row < input.rows(); row++) {
			input_padded(row + (int) df.y(), col + (int) df.x()) = input(row, col);
		}
	}

	// Save profiling data
	std::vector<double> tfunctional_runtimes(tfunctionals.size());
	std::vector<double> pfunctional_runtimes(pfunctionals.size(), 0);

	// Allocate a matrix for all output data to reside in
	Eigen::MatrixXd output(360 / ANGLE_INTERVAL, tfunctionals.size()*pfunctionals.size());

	// Process all T-functionals
	Profiler mainprofiler;
	if (vm.count("verbose"))
		std::cerr << "Calculating";
	for (size_t t = 0; t < tfunctionals.size(); t++) {
		// Calculate the trace transform sinogram
		if (vm.count("verbose"))
			std::cerr << " " << tfunctionals[t].name << "..." << std::flush;
		Profiler tprofiler;
		Eigen::MatrixXd sinogram = getTraceTransform(
			input_padded,
			ANGLE_INTERVAL,
			DISTANCE_INTERVAL,
			tfunctionals[t].wrapper
		);
		tprofiler.stop();
		tfunctional_runtimes[t] = tprofiler.elapsed();

		if (vm.count("debug") || pfunctionals.size() == 0) {
			// Save the sinogram image
			std::stringstream fn_trace_image;
			fn_trace_image << "trace_" << tfunctionals[t].name << ".pgm";
			pgmWrite(fn_trace_image.str(), mat2gray(sinogram));

			// Save the sinogram data
			std::stringstream fn_trace_data;
			fn_trace_data << "trace_" << tfunctionals[t].name << ".dat";
			dataWrite(fn_trace_data.str(), sinogram);
		}

		// Orthonormal functionals require the nearest orthonormal sinogram
		unsigned int sinogram_center;
		if (orthonormal)
			sinogram = nearest_orthonormal_sinogram(sinogram, sinogram_center);

		// Process all P-functionals
		for (size_t p = 0; p < pfunctionals.size(); p++) {
			// Extra parameters to functional
			// TODO std::bind
			if (orthonormal)
				dynamic_cast<HermiteFunctionalWrapper*>(pfunctionals[p].wrapper)
					->center(sinogram_center);

			// Calculate the circus function
			if (vm.count("verbose"))
				std::cerr << " " << pfunctionals[p].name << "..." << std::flush;
			Profiler pprofiler;
			Eigen::VectorXd circus = getCircusFunction(
				sinogram,
				pfunctionals[p].wrapper
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
			std::cerr << "t(" << tfunctionals[t].name << ") = "
				<< tfunctional_runtimes[t] << " s\n";
		}
		for (size_t p = 0; p < pfunctionals.size(); p++) {
			std::cerr << "t(" << pfunctionals[p].name << ") = "
				<< pfunctional_runtimes[p] / tfunctionals.size() << " s\n";
		}
	}

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
		dataWrite("circus.dat", output, headers);
	}

	return 0;
}
