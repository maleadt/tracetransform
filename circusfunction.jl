require("functionals")

function nearest_orthonormal_sinogram(input::Matrix)
        # Detect the offset of each column to the sinogram center
        sinogram_center = ifloor(rows(input) / 2.0);
        offset::Vector = Array(eltype(input), cols(input))
        for p in 1:cols
                median = find_weighted_median(input[p, :])
                offset[p] = median - sinogram_center
        end

        # Align each column to the sinogram center
        padding = max(offset) + abs(min(offset))
        new_center = sinogram_center + max(offset);
        aligned::Matrix = zeros(eltype(input), rows+padding, cols)
        for col in 1:cols(input)
                for row in 1:rows(input)
                        aligned[max+row-offset[col], col] = input[row, col]
                end
        end

        # Compute the nearest orthonormal sinogram
        (U, S, V) = svd(aligned)
        S = eye(size(S))
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
