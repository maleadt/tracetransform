using Images

immutable Point{T}
    i::T
    j::T
end

function imodf(x)
    (f, i) = modf(x)
    return f, itrunc(i)
end

function interpolate(source::AbstractImage{Float32,2}, p::Point{Float32})
    # Get fractional and integral part of the coordinates
    (i_fract, i_int) = imodf(p.i)
    (j_fract, j_int) = imodf(p.j)

    # Bilinear interpolation
    @inbounds return (source[i_int,   j_int]   * (1-j_fract) * (1-i_fract) +
                      source[i_int,   j_int+1] * j_fract     * (1-i_fract) +
                      source[i_int+1, j_int]   * (1-j_fract) * i_fract +
                      source[i_int+1, j_int+1] * j_fract     * i_fract)
end

function resize(input::AbstractImage{Float32,2}, new_size::(Uint, Uint))
    # Calculate transform matrix
    transform::Matrix = [
    size(input, 1)/new_size[1]  0;
    0                           size(input, 2)/new_size[2]];

    # Allocate output matrix
    # FIXME: zeros not necessary if we properly handle borders
    output::Matrix = zeros(new_size)
    
    # Process all points
    # FIXME: borders are wrong (but this doesn't matter here since we
    #        only handle padded images)
    for i in 2:new_size[1]-1
        for j in 2:new_size[2]-1
            # TODO: RowVector
            p::Matrix = [i j]
            p += [0.5 0.5]
            p *= transform
            p -= [0.5 0.5]

            # FIXME: this discards edge pixels
            if 1 <= p[1] < size(input, 1) && 1 <= p[2] < size(input, 2)
                output[i, j] = interpolate(input, p[1], p[2])
            end
        end
    end

    share(input, output)
end

function pad(input::AbstractImage{Float32,2})
    origin::Vector = ifloor(([size(input)...] .+ 1) ./ 2)
    rLast::Int = iceil(hypot(([size(input)...] .- 1 - origin)...)) + 1
    rFirst::Int = -rLast
    nBins::Int = rLast - rFirst + 1
    padded::Array{Float32} = zeros(nBins, nBins)
    origin_padded::Vector = ifloor(([size(padded)...] .+ 1) ./ 2)
    offset::Vector = origin_padded - origin
    endpoint::Vector = offset+([size(input)...])
    padded[1+offset[1]:endpoint[1], 1+offset[2]:endpoint[2]] = input.data

    share(input, padded)
end

function rotate(input::AbstractImage{Float32,2}, origin::Point{Float32},
                angle::Float32)
    if isyfirst(input)
        angle = -angle
    end

    # Calculate part of transform matrix
    const angle_cos = cosd(angle)
    const angle_sin = sind(angle)

    # Allocate output matrix
    output = zeros(input.data)

    # Process all pixels
    s = Point(size(input, 1), size(input, 2))
    for i in 1:s.i
        for j in 1:s.j
            # Get the source pixel
            t = Point(i-origin.i, j-origin.j)
            r = Point(-t.j*angle_sin + t.i*angle_cos + origin.i,
                       t.j*angle_cos + t.i*angle_sin + origin.j)

            # Copy if within bounds
            if 1 <= r.i < s.i && 1 <= r.j < s.j
                @inbounds output[i, j] = interpolate(input, r)
            end
        end
    end

    share(input, output)
end
