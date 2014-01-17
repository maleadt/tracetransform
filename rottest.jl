#!/usr/bin/julia

require("geometry")
require("auxiliary")

using Images

function main(args::Vector{Any})
    # Manage parameters
    if length(args) != 2
        error("Please provide input filename and rotation angle")
    end
    filename = args[1]
    angle = uint(args[2])

    # Read image
    input::Image{Float32} = gray2mat(imread(filename))
    printmat(int(255*input))

    # Rotate image
    origin::Vector{Float32} = floor(([size(input)...] .+ 1) ./ 2)
    input_rotated::Image{Float32} = rotate(input, origin, angle)

    # Output image
    print("\n")
    output = mat2gray(input_rotated)
    printmat(output)
end

function printmat(input)
    for row in 1:size(input, "y")
        for col in 1:size(input, "x")
            pixel = input["y", row, "x", col]
            if pixel == 0
                decimals = 1
            else
                decimals = ifloor(log10(pixel))+1
            end
            print("$pixel$(" "^(4-decimals))")
        end
        print("\n")
    end
end

main(ARGS)
