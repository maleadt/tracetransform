#!/usr/local/env julia

require("geometry")
require("auxiliary")

using Images

function main(args)
    # Manage parameters
    if length(args) != 2
        error("Please provide input filename and rotation angle")
    end
    filename = args[1]
    angle = float(args[2])

    # Read image
    input::Image{Float32} = gray2mat(imread(filename))
    printmat(mat2gray(input))

    # Rotate image
    origin::Point{Float32} = Point(floor(([size(input)...] .+ 1) ./ 2)...)
    output = rotate(input, origin, angle)

    # Output image
    print("\n")
    printmat(mat2gray(output))
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
