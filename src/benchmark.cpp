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
                _image = gray2mat(readpgm("res/Cam1_V1.pgm"));

                // Set-up the transformer
                _transformer = new Transformer(_image, false);
        }

        virtual void TearDown()
        {
                delete _transformer;
        }

        Transformer *_transformer;

private:
        Eigen::MatrixXf _image;
};

BENCHMARK_F(SmallImageFixture, Radon, 2, 3)
{
        std::vector<TFunctionalWrapper> tfunctionals{
                TFunctionalWrapper("radon",  TFunctional::Radon)
        };
        std::vector<PFunctionalWrapper> pfunctionals{

        };
        _transformer->getTransform(tfunctionals, pfunctionals);
}

BENCHMARK_F(SmallImageFixture, TraceRegular, 2, 3)
{
        std::vector<TFunctionalWrapper> tfunctionals{
                TFunctionalWrapper("T1",  TFunctional::T1)
        };
        std::vector<PFunctionalWrapper> pfunctionals{
                PFunctionalWrapper("P1",  PFunctional::P1)
        };
        _transformer->getTransform(tfunctionals, pfunctionals);
}


//
// Small orthonormal image test
//

class SmallOrthonormalImageFixture : public Hayai::Fixture
{
public:
        virtual void SetUp()
        {
                // Read the image
                _image = gray2mat(readpgm("res/Cam1_V1.pgm"));

                // Set-up the transformer
                _transformer = new Transformer(_image, true);
        }

        virtual void TearDown()
        {
                delete _transformer;
        }

        Transformer *_transformer;

private:
        Eigen::MatrixXf _image;
};

BENCHMARK_F(SmallOrthonormalImageFixture, TraceOrthonormal, 2, 3)
{
        std::vector<TFunctionalWrapper> tfunctionals{
                TFunctionalWrapper("T1",  TFunctional::T1)
        };
        std::vector<PFunctionalWrapper> pfunctionals{
                PFunctionalWrapper("H1",  PFunctional::Hermite, PFunctionalArguments(1))
        };
        _transformer->getTransform(tfunctionals, pfunctionals);
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
