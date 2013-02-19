require("functionals")

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
        functional::Functional)

        # Allocate the output matrix
        output::Vector = Array(
                eltype(input),
                cols(input)
        )

        # Trace all columns
        for p in 1:cols(input)
                output[p] = call(functional, vec(input[:, p]))
        end

        return output
end
