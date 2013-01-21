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

function interpolate(input::Matrix, p::Vector)
        # Get fractional and integral part of the coordinates
        integral::Vector = itrunc(p)
        fractional::Vector = p - integral

        # Bilinear interpolation
        return    input[integral[2],   integral[1]]   * (1-fractional[1]) * (1-fractional[2]) + 
                  input[integral[2],   integral[1]+1] * fractional[1]     * (1-fractional[2]) + 
                  input[integral[2]+1, integral[1]]   * (1-fractional[1]) * fractional[2] + 
                  input[integral[2]+1, integral[1]+1] * fractional[1]     * fractional[2]
end

function rotate(input::Matrix, origin::Vector, angle)
        # Calculate transform matrix
        # TODO: -angle
        transform::Matrix = [
                cosd(-angle) -sind(-angle);
                sind(-angle)  cosd(-angle)];

        # Allocate output matrix
        output::Matrix = zeros(
                eltype(input),
                size(input)...)

        # Process all points
        rows = size(input, 1)
        cols = size(input, 2)
        for col in 1:cols
                for row in 1:rows
                        # TODO: RowVector
                        p::Matrix = [col row]
                        # TODO: why no pixel center offset?
                        p -= origin'
                        p *= transform
                        p += origin'

                        # FIXME: this discards edge pixels
                        if 1 <= p[1] < cols && 1 <= p[2] < rows
                                output[row, col] = interpolate(input, vec(p))
                        end
                end
        end

        return output
end
