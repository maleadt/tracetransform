include("functionals.jl")
include("geometry.jl")
include("enum.jl")

@enum TFunctional Radon T1 T2 T3 T4 T5

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
        else
            error("unknown T-functional")
        end
        push!(tfunctionals, wrapper)
    end

    return tfunctionals
end

function getSinogram(input::Image{Float64}, angle_stepsize::Uint,
                     tfunctional::TFunctionalWrapper)
    @assert length(size(input)) == 2
    @assert size(input, 1) == size(input, 2)        # Padded image!

    # Get the image origin to rotate around
    origin::Vector{Float64} = floor(([size(input)...] .+ 1) ./ 2)

    # Allocate the output matrix
    output = similar(input, (size(input, "y"), ifloor(360/angle_stepsize)))
    output.properties["spatialorder"] = ["y", "x"]

    # Process all angles
    for a in 0:angle_stepsize:359
        # Rotate the image
        input_rotated::Image{Float64} = rotate(input, origin, a)

        # Process all projection bands
        a_index::Uint = a / angle_stepsize + 1
        for p in 1:size(input, "x")-1
            if tfunctional.functional == Radon
                output.data[p, a_index] = t_radon(slice(input_rotated, "x", p))
            elseif tfunctional.functional == T1
                output.data[p, a_index] = t_1(slice(input_rotated, "x", p))
            elseif tfunctional.functional == T2
                output.data[p, a_index] = t_2(slice(input_rotated, "x", p))
            elseif tfunctional.functional == T3
                output.data[p, a_index] = t_3(slice(input_rotated, "x", p))
            elseif tfunctional.functional == T4
                output.data[p, a_index] = t_4(slice(input_rotated, "x", p))
            elseif tfunctional.functional == T5
                output.data[p, a_index] = t_5(slice(input_rotated, "x", p))
            end
        end
    end

    return output
end
