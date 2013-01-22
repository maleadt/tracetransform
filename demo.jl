require("image")
require("geometry")
require("tracetransform")
require("auxiliary")

using ArgParse

abstract Functional

type TFunctional <: Functional
        functional::Function
end

type PFunctional <: Functional
        functional::Function
end

function main(args)
        # Parse the command arguments
        s = ArgParseSettings("Allowed options")
        @add_arg_table s begin
                "--verbose"
                        action = :store_true
                        help = "display some more details"
                "--debug"
                        action = :store_true
                        help = "write debug information"
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
                "output"
                        help = "where to write the output circus data"
                        default = "circus.dat"
        end
        parsed_args = parse_args(args, s)

        # Parse the functionals
        tfunctionals::Vector = TFunctional[]
        for functional in parsed_args["t-functional"]
                if functional == "0"
                        push!(tfunctionals, TFunctional(t_radon))
                else
                        error("unknown T-functional")
                end
        end
        pfunctionals::Vector = PFunctional[]
        for functional in parsed_args["p-functional"]
                error("unknown P-functional")
        end

        # Read the image
        const input = imread(parsed_args["input"])

        # Pad the image so we can freely rotate without losing information
        origin::Vector = ifloor(flipud([size(input)...] .+ 1) ./ 2)
        rLast::Int = iceil(hypot(([size(input)...] .- 1 - origin)...)) + 1
        rFirst::Int = -rLast
        nBins::Int = rLast - rFirst + 1
        input_padded::Array = zeros(eltype(input), nBins, nBins)
        origin_padded::Vector = ifloor(flipud([size(input_padded)...] .+ 1) ./ 2)
        offset::Vector = origin_padded - origin
        endpoint::Vector = offset+flipud([size(input)...])
        input_padded[1+offset[2]:endpoint[2], 1+offset[1]:endpoint[1]] = input

        # Calculate the sampling resolution
        angles::Vector = [0:1:360]
        diagonal = hypot(size(input)...)
        distances::Vector = [1:1:itrunc(diagonal)]

        # Process all T-functionals
        for tfunctional in tfunctionals
                # Calculate the trace transform sinogram
                const sinogram = getTraceTransform(
                        input_padded,
                        angles,
                        distances,
                        tfunctional.functional
                )

                if parsed_args["debug"]
                        # Save the sinogram image
                        imwrite("$tfunctional.pgm", sinogram)

                        # Save the sinogram data
                        dataWrite("$tfunctional.dat", sinogram);
                end

        end
end

main(ARGS)
