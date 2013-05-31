using Images

# TODO: generalize these algorithms, or assert2d/assertgray them

rows(input::Image) = size(input, 1)
cols(input::Image) = size(input, 2)

function interpolate(input::Image{Float64}, x::Float64, y::Float64)
    # Get fractional and integral part of the coordinates
    (x_fract, x_int) = modf(x)
    (y_fract, y_int) = modf(y)

    # Bilinear interpolation
    return   (input.data[y_int,   x_int]   * (1-x_fract) * (1-y_fract) + 
              input.data[y_int,   x_int+1] * x_fract     * (1-y_fract) + 
              input.data[y_int+1, x_int]   * (1-x_fract) * y_fract + 
              input.data[y_int+1, x_int+1] * x_fract     * y_fract)
end

function resize(input::Image{Float64}, new_rows::Uint, new_cols::Uint)
        # Calculate transform matrix
        transform::Matrix = [
                rows(input)/new_rows    0;
                0                       cols(input)/new_cols];

        # Allocate output matrix
        # FIXME: zeros not necessary if we properly handle borders
        output::Matrix = zeros(
                Float64,
                new_rows, new_cols)
        
        # Process all points
        # FIXME: borders are wrong (but this doesn't matter here since we
        #        only handle padded images)
        for col in 2:new_cols-1
                for row in 2:new_rows-1
                        # TODO: RowVector
                        p::Matrix = [col row]
                        p += [0.5 0.5]
                        p *= transform
                        p -= [0.5 0.5]

                        # FIXME: this discards edge pixels
                        if 1 <= p[1] < cols(input) && 1 <= p[2] < rows(input)
                            output[row, col] = interpolate(input, p[1], p[2])
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
    for col in 1:cols(input)
        for row in 1:rows(input)
            # Get the source pixel
            # FIXME: this was a nice matrix multiplication before, but Julia
            #        can't manage these small-matrix multiplications (issue 3239)
            xt::Float64 = col - origin[1]
            yt::Float64 = row - origin[2]
            x::Float64 =  xt*angle_cos + yt*angle_sin + origin[1]
            y::Float64 = -xt*angle_sin + yt*angle_cos + origin[2]

            # Copy if within bounds
            if 1 <= x < cols(input) && 1 <= y < rows(input)
                output[row, col] = interpolate(input, x, y)
            end
        end
    end

    share(input, output)
end
