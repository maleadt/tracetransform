using Images

# TODO: generalize these algorithms, or assert2d/assertgray them

rows(input::Image) = size(input, 1)
cols(input::Image) = size(input, 2)

@deprecate  rows(A)     size(A, 1)
@deprecate  cols(A)     size(A, 2)

function interpolate(input::Image{Float64}, i::Float64, j::Float64)
    # Get fractional and integral part of the coordinates
    (j_fract, j_int) = modf(j)
    (i_fract, i_int) = modf(i)

    # Bilinear interpolation
    return   (input.data[i_int,   j_int]   * (1-j_fract) * (1-i_fract) +
              input.data[i_int,   j_int+1] * j_fract     * (1-i_fract) +
              input.data[i_int+1, j_int]   * (1-j_fract) * i_fract +
              input.data[i_int+1, j_int+1] * j_fract     * i_fract)
end

function resize(input::Image{Float64}, new_size::(Uint, Uint))
        @assert length(size(input)) == 2
        # TODO: extract colordim? use wrapper for each cd?
        @assert length(new_size) == 2

        # Calculate transform matrix
        transform::Matrix = [
                size(input, 1)/new_size[1]  0;
                0                           size(input, 2)/new_size[2]];

        # Allocate output matrix
        # FIXME: zeros not necessary if we properly handle borders
        output::Matrix = zeros(
                Float64,
                new_size)
        
        # Process all points
        # FIXME: borders are wrong (but this doesn't matter here since we
        #        only handle padded images)
        for i in 2:new_size[1]-1
                for j in 2:new_size[2]-1
                        # TODO: RowVector
                        p::Matrix = [j i]
                        p += [0.5 0.5]
                        p *= transform
                        p -= [0.5 0.5]

                        # FIXME: this discards edge pixels
                        if 1 <= p[1] < size(input, 2) && 1 <= p[2] < size(input, 1)
                            output[i, j] = interpolate(input, p[2], p[1])
                        end
                end
        end

        share(input, output)
end

function pad(input::Image{Float64})
        origin::Vector = ifloor(flipud([size(input)...] .+ 1) ./ 2)
        rLast::Int = iceil(hypot(([size(input)...] .- 1 - origin)...)) + 1
        rFirst::Int = -rLast
        nBins::Int = rLast - rFirst + 1
        padded::Array = zeros(Float64, nBins, nBins)
        origin_padded::Vector = ifloor(flipud([size(padded)...] .+ 1) ./ 2)
        offset::Vector = origin_padded - origin
        endpoint::Vector = offset+flipud([size(input)...])
        padded[1+offset[2]:endpoint[2], 1+offset[1]:endpoint[1]] = input.data

        share(input, padded)
end

function rotate(input::Image{Float64}, origin::Vector{Float64}, angle::Real)
    # Calculate part of transform matrix
    angle_cos = cosd(-angle)
    angle_sin = sind(-angle)

    # Allocate output matrix
    output::Matrix{Float64} = zeros(
        Float64,
        size(input)...)

    # Process all pixels
    # TODO: swap 2 1
    for col in 1:size(input, 2)
        for row in 1:size(input, 1)
            # Get the source pixel
            # FIXME: this was a nice matrix multiplication before, but Julia
            #        can't manage these small-matrix multiplications (issue 3239)
            xt::Float64 = col - origin[1]
            yt::Float64 = row - origin[2]
            x::Float64 =  xt*angle_cos + yt*angle_sin + origin[1]
            y::Float64 = -xt*angle_sin + yt*angle_cos + origin[2]

            # Copy if within bounds
            if 1 <= x < size(input, 2) && 1 <= y < size(input, 1)
                output[row, col] = interpolate(input, y, x)
            end
        end
    end

    share(input, output)
end
