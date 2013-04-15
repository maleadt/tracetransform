//
// Configuration
//

// Standard library
#include <vector>
#include <exception>

// Hayai
#include "../lib/hayai/src/hayai.hpp"

// Boost
#include <boost/program_options.hpp>

// Local
#include "logger.hpp"
#include "auxiliary.hpp"
extern "C" {
        #include "functionals.h"
}
#include "wrapper.hpp"
#include "transform.hpp"


//
// Small image test
//

class SmallImageFixture : public Hayai::Fixture
{
public:
        virtual void SetUp()
        {
                // Read the image
                _image = pgmRead("res/Cam1_V1.pgm");
                _image = gray2mat(_image);

                // Set-up the transformer
                _transformer = new Transformer(_image);
        }

        virtual void TearDown()
        {
                delete _transformer;
        }

        Transformer *_transformer;

private:
        Eigen::MatrixXd _image;
};

BENCHMARK_F(SmallImageFixture, Radon, 2, 3)
{
        std::vector<TFunctional> tfunctionals{
                TFunctional("Radon",  new SimpleFunctionalWrapper(TFunctionalRadon))
        };
        std::vector<PFunctional> pfunctionals{

        };
        Eigen::MatrixXd output = _transformer->getTransform(tfunctionals, pfunctionals);
}

BENCHMARK_F(SmallImageFixture, TraceRegular, 2, 3)
{
        std::vector<TFunctional> tfunctionals{
                TFunctional("T1",  new SimpleFunctionalWrapper(TFunctional1))
        };
        std::vector<PFunctional> pfunctionals{
                PFunctional("P1",  new SimpleFunctionalWrapper(PFunctional1))
        };
        Eigen::MatrixXd output = _transformer->getTransform(tfunctionals, pfunctionals);
}

BENCHMARK_F(SmallImageFixture, TraceOrthonormal, 2, 3)
{
        std::vector<TFunctional> tfunctionals{
                TFunctional("T1",  new SimpleFunctionalWrapper(TFunctional1))
        };
        std::vector<PFunctional> pfunctionals{
                PFunctional("H1",
                                new GenericFunctionalWrapper<unsigned int, size_t>(PFunctionalHermite),
                                PFunctional::Type::HERMITE,
                                1)
        };
        Eigen::MatrixXd output = _transformer->getTransform(tfunctionals, pfunctionals);
}


//
// Main application
//

int main(int argc, char **argv)
{
        //
        // Initialization
        //

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
        ;

        // Parse the options
        boost::program_options::variables_map vm;
        try {
                store(boost::program_options::command_line_parser(argc, argv)
                        .options(desc).run(), vm);
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
        // Benchmarking
        //

        Hayai::Benchmarker::RunAllTests();
        return 0;
}
