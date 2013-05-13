require("functionals")
require("enum")

@enum PFunctional NoP Hermite P1 P2 P3

type PFunctionalArguments
        order::Uint
        center::Uint

        function PFunctionalArguments()
                order = 0
                center = 0
                new(functional, arguments)
        end
end

type PFunctionalWrapper
        functional::PFunctional
        arguments::PFunctionalArguments

        function PFunctionalWrapper()
                functional = NoP
                arguments = PFunctionalArguments()
                new(functional, arguments)
        end
end

function parse_pfunctionals(args)
        pfunctionals::Vector = PFunctionalWrapper[]
        for functional in args
                wrapper = PFunctionalWrapper
                if functional == "1"
                        wrapper.functional = P1
                elseif functional == "2"
                        wrapper.functional = P2
                elseif functional == "3"
                        wrapper.functional = P3
                elseif functional[1] == 'H'
                        wrapper.functional = Hermite
                        wrapper.arguments.order = parse_int(functional[2:])
                else
                        error("unknown P-functional")
                end
                push!(tfunctionals, wrapper)
        end

        return pfunctionals
end

function nearest_orthonormal_sinogram(input::Matrix)
        # Detect the offset of each column to the sinogram center
        sinogram_center = ifloor(rows(input) / 2.0);
        offset::Vector = Array(eltype(input), cols(input))
        for p in 1:cols(input)
                median = find_weighted_median(vec(input[p, :]))
                offset[p] = median - sinogram_center
        end

        # Align each column to the sinogram center
        padding::Uint = max(offset) + abs(min(offset))
        new_center = sinogram_center + max(offset);
        aligned::Matrix = zeros(eltype(input), rows(input)+padding, cols(input))
        for col in 1:cols(input)
                for row in 1:rows(input)
                        aligned[max(offset)+row-offset[col], col] = input[row, col]
                end
        end

        # Compute the nearest orthonormal sinogram
        (U, _, V) = svd(aligned)
        S = eye(size(U, 2), size(V, 1))
        nos = U*S*V'

        return (sinogram_center, nos)
end

function getCircusFunction(
        input::Matrix,
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
