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
        # Detect the offset of each column to the sinogram center
        sinogram_center = ifloor(rows(input) / 2.0);
        offset::Vector = Array(eltype(input), cols(input))
        for p in 1:cols(input)
                median = find_weighted_median(vec(input.data[p, :]))
                offset[p] = median - sinogram_center
        end

        # Align each column to the sinogram center
        padding::Uint = max(offset) + abs(min(offset))
        new_center = sinogram_center + max(offset);
        aligned::Matrix = zeros(Float64, rows(input)+padding, cols(input))
        for col in 1:cols(input)
                for row in 1:rows(input)
                        aligned[max(offset)+row-offset[col], col] = input.data[row, col]
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
                eltype(input),
                cols(input)
        )

        # Trace all columns
        for p in 1:cols(input)
                if pfunctional.functional == P1
                        output[p] = p_1(vec(input[:, p]))
                elseif pfunctional.functional == P2
                        output[p] = p_2(vec(input[:, p]))
                elseif pfunctional.functional == P3
                        output[p] = p_3(vec(input[:, p]))
                elseif pfunctional.functional == Hermite
                        output[p] = p_hermite(vec(input[:, p]), pfunctional.arguments.order, pfunctional.arguments.center)
                end
        end

        return output
end
