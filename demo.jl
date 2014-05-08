#!/usr/bin/julia

require("geometry")
require("transform")
require("auxiliary")
require("enum")
require("log")

using ArgParse
using Images

function main(args)
    #
    # Initialization
    #
    
    # Parse the command arguments
    s = ArgParseSettings("Allowed options")
    s.autofix_names = true
    @add_arg_table s begin
        "--quiet", "-q"
            help = "only display errors and warnings"
            action = :store_true
        "--verbose", "-v"
            help = "display some more details"
            action = :store_true
        "--debug", "-d"
            help = "write debug information"
            action = :store_true
        "--t-functional", "-T"
            help = "T-functionals"
            action = :append_arg
            arg_type = String
        "--p-functional", "-P"
            help = "P-functionals"
            action = :append_arg
            arg_type = String
        "--mode", "-m"
            help = "execution mode (calculate or benchmark)"
            arg_type = String
        "--iterations", "-n"
            help = "amount of iterations to benchmark"
            arg_type = Uint
        "--angle", "-a"
            help = "angle stepsize"
            arg_type = Uint
            default = uint(1)
        "input"
            help = "image to process"
            required = true
    end
    opts = parse_args(args, s)

    # Check for required named arguments
    # FIXME: ArgParse only allows requiring positional arguments
    if opts["mode"] == nothing
        error("required argument mode was not provided")
    end
    if opts["t_functional"] == nothing
        error("required argument t-functional was not provided")
    end

    # Parse the functionals
    tfunctionals::Vector{TFunctionalWrapper} =
        parse_tfunctionals(opts["t_functional"])
    pfunctionals::Vector{PFunctionalWrapper} =
        parse_pfunctionals(opts["p_functional"])

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
    input::Image{Float32} = gray2mat(imread(opts["input"]))
    if isxfirst(input)
        # Since we only scan columns, make sure the image is laid out column major
        println("Transposing")
        input = ctranspose(input)
    end
    (input, basename) = prepare_transform(input, opts["input"], opts["angle"], orthonormal)

    if opts["mode"] == "calculate"
        get_transform(input, basename, tfunctionals, pfunctionals,
                      opts["angle"], orthonormal, true)
    elseif opts["mode"] == "benchmark"
        if opts["iterations"] == nothing
            error("required argument iterations was not provided")
        end

        # Warm-up heavily
        for i = 1:3
            get_transform(input, basename, tfunctionals, pfunctionals,
                          opts["angle"], orthonormal, false)
        end

        for i = 1:opts["iterations"]
            time = @elapsed get_transform(input, basename, tfunctionals,
                                          pfunctionals, opts["angle"],
                                          orthonormal, false)
            println("t_$(i)=$(time)")
        end
    elseif opts["mode"] == "profile"
        require("ProfileView.jl")

        # Warm-up
        get_transform(input, basename, tfunctionals, pfunctionals,
                      opts["angle"], orthonormal, false)

        @profile get_transform(input, basename, tfunctionals,
                              pfunctionals, opts["angle"],
                              orthonormal, false)
        Profile.print()
        ProfileView.view()
        while true
            sleep(10)
        end
    else
        error("invalid execution mode")
    end
end

if !isinteractive()
    main(ARGS)
end
