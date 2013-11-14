require("functionals")
require("enum")

@enum PFunctional Hermite P1 P2 P3

type PFunctionalArguments
    order::Uint
    center::Uint

    function PFunctionalArguments()
        order = 0
        center = 0
        new(order, center)
    end
end

type PFunctionalWrapper
    functional::PFunctional
    arguments::PFunctionalArguments

    PFunctionalWrapper(functional::PFunctional) =
    new(functional, PFunctionalArguments())
end

function parse_pfunctionals(args::Vector{String})
    pfunctionals::Vector{PFunctionalWrapper} = PFunctionalWrapper[]
    for functional in args
        if functional == "1"
            wrapper = PFunctionalWrapper(P1)
        elseif functional == "2"
            wrapper = PFunctionalWrapper(P2)
        elseif functional == "3"
            wrapper = PFunctionalWrapper(P3)
        elseif functional[1] == 'H'
            wrapper = PFunctionalWrapper(Hermite)
            wrapper.arguments.order = parseint(functional[2:])
        else
            error("unknown P-functional")
        end
        push!(pfunctionals, wrapper)
    end

    return pfunctionals
end

function nearest_orthonormal_sinogram(input::Image{Float64})
    @assert length(size(input)) == 2

    cols = size(input, "x")
    rows = size(input, "y")

    # Detect the offset of each column to the sinogram center
    sinogram_center = ifloor(rows / 2);
    offset::Vector = Array(Float64, cols)
    for p in 1:cols
        median = find_weighted_median(input["y", p])
        offset[p] = median - sinogram_center
    end

    # Align each column to the sinogram center
    padding::Uint = maximum(offset) + abs(mimimum(offset))
    new_center = sinogram_center + maximum(offset);
    aligned::Matrix = zeros(rows+padding, cols)
    for col in 1:cols
        for row in 1:rows
            aligned[maximum(offset)+row-offset[col], col] =
            input.data["x", col, "y", row]
        end
    end

    # Compute the nearest orthonormal sinogram
    (U, _, V) = svd(aligned)
    S = eye(size(U, 2), size(V, 1))
    nos = U*S*V'

    return (sinogram_center, share(input, nos))
end

function getCircusFunction(
    input::Image{Float64},
    pfunctional::PFunctionalWrapper)

    # Allocate the output matrix
    output::Vector = Array(
        Float64,
        size(input, "x")
        )

    # Trace all columns
    for p in 1:size(input, "x")
        if pfunctional.functional == P1
            output[p] = p_1(slice(input, "x", p))
        elseif pfunctional.functional == P2
            output[p] = p_2(slice(input, "x", p))
        elseif pfunctional.functional == P3
            output[p] = p_3(slice(input, "x", p))
        elseif pfunctional.functional == Hermite
            output[p] = p_hermite(slice(input, "x", p),
              pfunctional.arguments.order,
              pfunctional.arguments.center)
        end
    end

    return output
end
