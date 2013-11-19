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
        "--p-functional", "-P"
            action = :append_arg
            help = "P-functionals"
            arg_type = String
        "--mode", "-m"
            help = "execution mode (calculate or benchmark)"
            arg_type = String
        "--angle", "-a"
            help = "angle stepsize"
            arg_type = Uint
            default = uint(1)
        "--iterations", "-n"
            help = "amount of iterations to run"
            arg_type = Uint
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
    if opts["t-functional"] == nothing
        error("required argument t-functional was not provided")
    end

    # Parse the functionals
    tfunctionals::Vector{TFunctionalWrapper} =
        parse_tfunctionals(opts["t-functional"])
    pfunctionals::Vector{PFunctionalWrapper} =
        parse_pfunctionals(opts["p-functional"])

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
    input = prepare_transform(input, opts["angle"], orthonormal)

    if opts["mode"] == "calculate"
        get_transform(input, tfunctionals, pfunctionals, opts["angle"],
                      orthonormal, true)

    elseif opts["mode"] == "benchmark"
        if opts["iterations"] == nothing
            error("required argument iterations was not provided")
        end

        for i = 1:opts["iterations"]
            time = @elapsed get_transform(input, tfunctionals, pfunctionals,
                                          opts["angle"], orthonormal, false)
            println("t_$(i)=$(time)")
        end
    else
        error("invalid execution mode")
    end
end

main(ARGS)
