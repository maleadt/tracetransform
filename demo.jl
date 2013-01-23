require("image")
require("geometry")
require("tracetransform")
require("circusfunction")
require("auxiliary")
require("functionals")

using ArgParse

ANGLE_INTERVAL = 3
DISTANCE_INTERVAL = 1

function main(args)
        #
        # Initialization
        #
        
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
        tfunctionals::Vector = Functional[]
        for functional in parsed_args["t-functional"]
                if functional == "0"
                        push!(tfunctionals, SimpleFunctional(t_radon, "T0"))
                elseif functional == "1"
                        push!(tfunctionals, SimpleFunctional(t_1, "T1"))
                else
                        error("unknown T-functional")
                end
        end
        pfunctionals::Vector = Functional[]
        for functional in parsed_args["p-functional"]
                if functional == "1"
                        push!(pfunctionals, SimpleFunctional(p_1, "P1"))
                else
                        error("unknown P-functional")
                end
        end

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


        #
        # Image processing
        #

        # Read the image
        input = imread(parsed_args["input"])

        # Orthonormal P-functionals need a stretched image in order to ensure a
        # square sinogram
        if orthonormal
                ndiag = iceil(360.0 / ANGLE_INTERVAL)
                nsize = iceil(ndiag / sqrt(2))
                input = resize(input, nsize, nsize)
        end

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
        angles::Vector = [0:ANGLE_INTERVAL:360] # FIXME: itrunc vs iceil?
        diagonal = hypot(size(input)...)
        distances::Vector = [1:DISTANCE_INTERVAL:itrunc(diagonal)]

        # Allocate a matrix for all output data to reside in
        output::Matrix = Array(
                eltype(input),
                length(angles),
                length(tfunctionals) * length(pfunctionals))

        # Process all T-functionals
        t_i = 0
        print("Calculating")
        for tfunctional in tfunctionals
                # Calculate the trace transform sinogram
                print(" $(tfunctional.name)...")
                const sinogram = getTraceTransform(
                        input_padded,
                        angles,
                        distances,
                        tfunctional
                )
                
                if parsed_args["debug"]
                        # Save the sinogram image
                        ppmwrite(mat2gray(sinogram), "trace_$(tfunctional.name).pgm")

                        # Save the sinogram data
                        datawrite("trace_$(tfunctional.name).dat", sinogram);
                end

                # Orthonormal functionals require the nearest orthonormal sinogram
                if orthonormal
                        (sinogram_center, sinogram) = nearest_orthonormal_sinogram(sinogram)
                end

                # Process all P-functionals
                p_i = 1
                for pfunctional in pfunctionals
                        # Configure any extra parameters
                        if isa(pfunctional, HermiteFunctional)
                                pfunctional.center = sinogram_center
                        end

                        # Calculate the circus function
                        print(" $(pfunctional.name)...")
                        circus::Vector = getCircusFunction(
                                sinogram,
                                pfunctional
                        )

                        # Normalize
                        normalized = zscore(circus)

                        # Copy the data
                        @assert length(normalized) == rows(output)
                        output[:, t_i*length(tfunctionals) + p_i] = normalized
                        p_i += 1
                end
                t_i += 1
        end
end

main(ARGS)
