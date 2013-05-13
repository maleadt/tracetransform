require("geometry")
require("transform")
require("auxiliary")
require("enum")

using ArgParse
using Images

@enum LogLevel trace debug info warning error fatal
VERBOSITY = info
function want_log(level::LogLevel)
        return level.n >= VERBOSITY.n
end


function main(args)
        #
        # Initialization
        #
        
        # Parse the command arguments
        s = ArgParseSettings("Allowed options")
        s.suppress_warnings = true # TODO: fix bug in argparse
        @add_arg_table s begin
                "--quiet", "-q"
                        action = :store_true
                        help = "only display errors and warnings"
                "--verbose", "-v"
                        action = :store_true
                        help = "display some more details"
                "--debug", "-d"
                        action = :store_true
                        help = "write even more details"
                "--t-functional", "-T"
                        action = :append_arg
                        help = "T-functionals"
                        # TODO: arg_type = TFunctional
                        # TODO; required = true
                "--p-functional", "-P"
                        action = :append_arg
                        help = "P-functionals"
                        # TODO: arg_type = PFunctional
                "input"
                        help = "image to process"
                        required = true
        end
        parsed_args = parse_args(args, s)

        # Parse the functionals
        tfunctionals::Vector = parse_tfunctionals(parsed_args["t-functional"])
        pfunctionals::Vector = parse_pfunctionals(parsed_args["p-functional"])

        # Check for orthonormal P-functionals
        orthonormal_count = 0
        for functional in pfunctionals
                if isa(functional, HermiteFunctional)
                        orthonormal_count += 1
                end
        end
        if orthonormal_count == 0
                orthonormal = false
        elseif orthonormal_count == length(pfunctionals)
                orthonormal = true
        else
                error("cannot mix regular and orthonormal P-functionals")
        end

        # Configure logging
        if parsed_args["verbose"]
                VERBOSITY = debug
        elseif parsed_args["debug"]
                VERBOSITY = trace
        elseif parsed_args["quiet"]
                VERBOSITY = warning
        end


        #
        # Image processing
        #

        # Read the image
        input_image = imread(parsed_args["input"])
        input = gray2mat(input_image)

        # Transform the image
        input = prepare_transform(input, orthonormal)
        get_transform(input, tfunctionals, pfunctionals, orthonormal)
end

main(ARGS)
