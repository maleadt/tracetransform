#!/usr/local/bin/julia

require("geometry")
require("transform")
require("auxiliary")
require("enum")
require("log")

using ArgParse
using Images

function main(args::Vector{Any})
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
                        arg_type = String
                        required = true
                "--p-functional", "-P"
                        action = :append_arg
                        help = "P-functionals"
                        arg_type = String
                "--mode", "-m"
                        help = "execution mode"
                        arg_type = String
                        required = true
                "--iterations", "-n"
                        help = "amount of iterations to run"
                        arg_type = Uint
                "input"
                        help = "image to process"
                        required = true
        end
        opts = parse_args(args, s)

        # Parse the functionals
        tfunctionals::Vector{TFunctionalWrapper} = parse_tfunctionals(opts["t-functional"])
        pfunctionals::Vector{PFunctionalWrapper} = parse_pfunctionals(opts["p-functional"])

        # Check for orthonormal P-functionals
        orthonormal_count = 0
        for pfunctional in pfunctionals
                if pfunctional.functional == Hermite
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
        if opts["verbose"]
                set_threshold(debug_l)
        elseif opts["debug"]
                set_threshold(trace_l)
        elseif opts["quiet"]
                set_threshold(warning_l)
        end


        #
        # Execution
        #

        # Read the image
        input::Image{Float64} = gray2mat(imread(opts["input"]))
        input = prepare_transform(input, orthonormal)

        if opts["mode"] == "calculate"
                get_transform(input, tfunctionals, pfunctionals, orthonormal, true)

        elseif opts["mode"] == "benchmark"

        else
                error("invalid execution mode")
        end
end

main(ARGS)
