require("image.jl")
require("geometry.jl")
require("tracetransform.jl")

function main ()
        # Check and read the parameters
        if length(ARGS) < 1
                error("Invalid usage: demo.jl IMAGE")
        end
        const fn_input = ARGS[1]

        # Read the image
        const input = imread(fn_input)

        # Pad the image so we can freely rotate without losing information
        origin::Vector = ifloor(([size(input)...] .+ 1) ./ 2)
        rLast::Int = iceil(hypot(([size(input)...] .- 1 - origin)...)) + 1
        rFirst::Int = -rLast
        nBins::Int = rLast - rFirst + 1
        input_padded::Array = zeros(eltype(input), nBins, nBins)
        origin_padded::Vector = ifloor(([size(input_padded)...] .+ 1) ./ 2)
        offset::Vector = origin_padded - origin
        endpoint::Vector = offset+[size(input)...]
        input_padded[1+offset[1]:endpoint[1], 1+offset[2]:endpoint[2]] = input

        # Calculate the sampling resolution
        angles::Vector{Real} = [0:1:360]
        diagonal = hypot(size(input)...)
        distances::Vector{Real} = [0:1:diagonal]

        # Get the trace transform
        const transform = getTraceTransform(
                input_padded,
                angles,
                distances
        )

        # Display or write the image
        if length(ARGS) < 2
                imshow(transform)
        else
                const fn_output = ARGS[2]
                imwrite(transform, fn_output)
        end
end

main()
