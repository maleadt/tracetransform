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

function pad(input::Image{Float64})
    @assert length(size(input)) == 2

    origin::Vector = ifloor(([size(input)...] .+ 1) ./ 2)
    rLast::Int = iceil(hypot(([size(input)...] .- 1 - origin)...)) + 1
    rFirst::Int = -rLast
    nBins::Int = rLast - rFirst + 1
    padded::Array{Float64} = zeros(nBins, nBins)
    origin_padded::Vector = ifloor(([size(padded)...] .+ 1) ./ 2)
    offset::Vector = origin_padded - origin
    endpoint::Vector = offset+([size(input)...])
    padded[1+offset[1]:endpoint[1], 1+offset[2]:endpoint[2]] = input.data

    share(input, padded)
end

function rotate(input::Image{Float64}, origin::Vector{Float64}, angle::Real)
    @assert length(size(input)) == 2

    # Calculate part of transform matrix
    angle_cos = cosd(angle)
    angle_sin = sind(angle)

    # Allocate output matrix
    output::Matrix{Float64} = zeros(size(input))

    # Process all pixels
    for i in 1:size(input, 1)
        for j in 1:size(input, 2)
            # Get the source pixel
            # FIXME: this was a nice matrix multiplication before, but Julia
            #        can't manage these small-matrix multiplications (issue 3239)
            i_t::Float64 = i - origin[1]
            j_t::Float64 = j - origin[2]
            i_r::Float64 = -j_t*angle_sin + i_t*angle_cos + origin[1]
            j_r::Float64 =  j_t*angle_cos + i_t*angle_sin + origin[2]

            # Copy if within bounds
            if 1 <= i_r < size(input, 1) && 1 <= j_r < size(input, 2)
                output[i, j] = interpolate(input, i_r, j_r)
            end
        end
    end

    share(input, output)
end
