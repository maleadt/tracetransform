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

function getSinograms(input::Image{Float64}, angle_stepsize::Uint,
                      tfunctionals::Vector{TFunctionalWrapper})
    @assert length(size(input)) == 2
    @assert size(input, 1) == size(input, 2)        # Padded image!

    # Get the image origin to rotate around
    origin::Vector{Float64} = floor(([size(input)...] .+ 1) ./ 2)

    # Allocate the output matrix
    outputs = Array(Image, length(tfunctionals))
    for t in 1:length(tfunctionals)
        outputs[t] = similar(input, (size(input, "y"), ifloor(360/angle_stepsize)))
        outputs[t].properties["spatialorder"] = ["y", "x"]
    end

    # TODO: precalculate

    # Process all angles
    for a in 0:angle_stepsize:359
        # Rotate the image
        input_rotated::Image{Float64} = rotate(input, origin, a)

        # Process all projection bands
        a_index::Uint = a / angle_stepsize + 1
        for p in 1:size(input, "x")-1
            data = slice(input_rotated, "x", p)

            # Process all T-functionals
            for t in 1:length(tfunctionals)
                if tfunctionals[t].functional == Radon
                    outputs[t].data[p, a_index] = t_radon(data)
                elseif tfunctionals[t].functional == T1
                    outputs[t].data[p, a_index] = t_1(data)
                elseif tfunctionals[t].functional == T2
                    outputs[t].data[p, a_index] = t_2(data)
                elseif tfunctionals[t].functional == T3
                    outputs[t].data[p, a_index] = t_3(data)
                elseif tfunctionals[t].functional == T4
                    outputs[t].data[p, a_index] = t_4(data)
                elseif tfunctionals[t].functional == T5
                    outputs[t].data[p, a_index] = t_5(data)
                end
            end
        end
    end

    return outputs
end
