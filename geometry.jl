using Images

# TODO: generalize these algorithms, or assert2d/assertgray them

rows(input::Image) = size(input, 1)
cols(input::Image) = size(input, 2)

function interpolate(input::Image, p::Vector)
        # Get fractional and integral part of the coordinates
        (x_fract, x_int) = modf(p[1])
        (y_fract, y_int) = modf(p[2])

        # Bilinear interpolation
        return   (input.data[y_int,   x_int]   * (1-x_fract) * (1-y_fract) + 
                  input.data[y_int,   x_int+1] * x_fract     * (1-y_fract) + 
                  input.data[y_int+1, x_int]   * (1-x_fract) * y_fract + 
                  input.data[y_int+1, x_int+1] * x_fract     * y_fract)
end

function resize(input::Image, new_rows, new_cols)
        # Calculate transform matrix
        transform::Matrix = [
                rows(input)/new_rows    0;
                0                       cols(input)/new_cols];

        # Allocate output matrix
        # FIXME: zeros not necessary if we properly handle borders
        output::Matrix = zeros(
                eltype(input),
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
                            output[row, col] = interpolate(input, vec(p))
                        end
                end
        end

        share(input, output)
end

function pad(input::Image)
        origin::Vector = ifloor(flipud([size(input)...] .+ 1) ./ 2)
        rLast::Int = iceil(hypot(([size(input)...] .- 1 - origin)...)) + 1
        rFirst::Int = -rLast
        nBins::Int = rLast - rFirst + 1
        padded::Array = zeros(eltype(input), nBins, nBins)
        origin_padded::Vector = ifloor(flipud([size(padded)...] .+ 1) ./ 2)
        offset::Vector = origin_padded - origin
        endpoint::Vector = offset+flipud([size(input)...])
        padded[1+offset[2]:endpoint[2], 1+offset[1]:endpoint[1]] = input.data

        share(input, padded)
end

function rotate(input::Image, origin::Vector, angle)
        # Calculate transform matrix
        const transform::Matrix = [
                cosd(-angle) -sind(-angle);
                sind(-angle)  cosd(-angle)];

        # Allocate output matrix
        output::Matrix = zeros(
                eltype(input),
                size(input)...)

        # Process all points
        for col in 1:cols(input)
                for row in 1:rows(input)
                        # TODO: RowVector
                        p::Matrix = [col row]
                        # TODO: why no pixel center offset?
                        p -= origin'
                        p *= transform
                        p += origin'

                        # FIXME: this discards edge pixels
                        if 1 <= p[1] < cols(input) && 1 <= p[2] < rows(input)
                                output[row, col] = interpolate(input, vec(p))
                        end
                end
        end

        share(input, output)
end
