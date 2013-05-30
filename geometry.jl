type Point2D{T} <: AbstractVector{T}
        x::T
        y::T
end

Point2D() = Point2D(0, 0)

import Base.convert
convert{T}(::Type{Point2D{T}}, x::AbstractVector{T}) = 
        Point2D{T}(x[1], x[2])
convert{T}(::Type{AbstractVector{T}}, p::Point2D{T}) =
        [p.x p.y]

rows(input::Matrix) = size(input, 1)
cols(input::Matrix) = size(input, 2)

function interpolate(input::Matrix, p::Vector)
        # Get fractional and integral part of the coordinates
        float x_int, y_int;
        float x_fract = std::modf(p.x(), &x_int);
        float y_fract = std::modf(p.y(), &y_int);
        integral::Vector = itrunc(p)
        fractional::Vector = p - integral

        # Bilinear interpolation
        return   (input[integral[2],   integral[1]]   * (1-fractional[1]) * (1-fractional[2]) + 
                  input[integral[2],   integral[1]+1] * fractional[1]     * (1-fractional[2]) + 
                  input[integral[2]+1, integral[1]]   * (1-fractional[1]) * fractional[2] + 
                  input[integral[2]+1, integral[1]+1] * fractional[1]     * fractional[2])
end

function resize(input::Matrix, new_rows, new_cols)
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

        return output
end

function pad(input::Matrix)
        origin::Vector = ifloor(flipud([size(input)...] .+ 1) ./ 2)
        rLast::Int = iceil(hypot(([size(input)...] .- 1 - origin)...)) + 1
        rFirst::Int = -rLast
        nBins::Int = rLast - rFirst + 1
        input_padded::Array = zeros(eltype(input), nBins, nBins)
        origin_padded::Vector = ifloor(flipud([size(input_padded)...] .+ 1) ./ 2)
        offset::Vector = origin_padded - origin
        endpoint::Vector = offset+flipud([size(input)...])
        input_padded[1+offset[2]:endpoint[2], 1+offset[1]:endpoint[1]] = input

        return input_padded
end

function rotate(input::Matrix, origin::Vector, angle)
        # Calculate transform matrix
        transform::Matrix = [
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

        return output
end
