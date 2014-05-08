require("functionals")
require("geometry")
require("enum")

using Images
using ArrayViews
import ArrayViews.view
view(img::AbstractImage, dimname::ASCIIString, ind::RangeIndex, nameind::RangeIndex...) = view(img.data, coords(img, dimname, ind, nameind...)...)

@enum TFunctional Radon T1 T2 T3 T4 T5 T6 T7

type TFunctionalArguments
end

type TFunctionalWrapper
    functional::TFunctional
    arguments::TFunctionalArguments

    TFunctionalWrapper(functional::TFunctional) =
    new(functional, TFunctionalArguments())
end

function parse_tfunctionals(args::Vector{String})
    tfunctionals::Vector{TFunctionalWrapper} = TFunctionalWrapper[]
    for functional in args
        if functional == "0"
            wrapper = TFunctionalWrapper(Radon)
        elseif functional == "1"
            wrapper = TFunctionalWrapper(T1)
        elseif functional == "2"
            wrapper = TFunctionalWrapper(T2)
        elseif functional == "3"
            wrapper = TFunctionalWrapper(T3)
        elseif functional == "4"
            wrapper = TFunctionalWrapper(T4)
        elseif functional == "5"
            wrapper = TFunctionalWrapper(T5)
        elseif functional == "6"
            wrapper = TFunctionalWrapper(T6)
        elseif functional == "7"
            wrapper = TFunctionalWrapper(T7)
        else
            error("unknown T-functional")
        end
        push!(tfunctionals, wrapper)
    end

    return tfunctionals
end

function getSinograms(input::AbstractImage{Float32,2}, angle_stepsize::Uint,
                      tfunctionals::Vector{TFunctionalWrapper})
    # Image should be padded
    @assert size(input, 1) == size(input, 2)

    # Get the image origin to rotate around
    origin::Point{Float32} = Point(float32(floor(([size(input)...] .+ 1) ./ 2))...)

    # Allocate the output matrix
    outputs = Array(Image{Float32}, length(tfunctionals))
    for t in 1:length(tfunctionals)
        outputs[t] = similar(input, (size(input, "y"), ifloor(360/angle_stepsize)))
        outputs[t].properties["spatialorder"] = ["y", "x"]
    end

    # Precalculate
    precalculations = Dict{TFunctional, Any}()
    precalculation_input = (size(input, "y"), size(input, "x"))
    for tfunctional in tfunctionals
        if tfunctional.functional == T3
            precalculations[T3] = t_3_prepare(Float32, precalculation_input...)
        end
        if tfunctional.functional == T4
            precalculations[T4] = t_4_prepare(Float32, precalculation_input...)
        end
        if tfunctional.functional == T5
            precalculations[T5] = t_5_prepare(Float32, precalculation_input...)
        end
    end

    # Process all angles
    t_total = 0
    for a in 0:angle_stepsize:359
        # Rotate the image
        input_rotated::Image{Float32} = rotate(input, origin, float32(a))

        # Process all projection bands
        a_index::Int = a / angle_stepsize + 1
        for p in 1:size(input, "x")-1   # TODO: -1?
            data = view(input_rotated, "x", p)

            # Process all T-functionals
            for t in 1:length(tfunctionals)
                if tfunctionals[t].functional == Radon
                    outputs[t].data[p, a_index] = t_radon(data)
                elseif tfunctionals[t].functional == T1
                    outputs[t].data[p, a_index] = t_1(data)
                elseif tfunctionals[t].functional == T2
                    outputs[t].data[p, a_index] = t_2(data)
                elseif tfunctionals[t].functional == T3 ||
                       tfunctionals[t].functional == T4 ||
                       tfunctionals[t].functional == T5
                    outputs[t].data[p, a_index] =
                        t_345(data, precalculations[tfunctionals[t].functional])
                elseif tfunctionals[t].functional == T6
                    outputs[t].data[p, a_index] = t_6(data)
                elseif tfunctionals[t].functional == T7
                    outputs[t].data[p, a_index] = t_7(data)
                end
            end
        end
    end

    return outputs
end
