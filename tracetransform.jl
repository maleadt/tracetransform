function getTraceTransform{T}(
	input::Matrix{T},
	angles::Vector,
	distances::Vector)
	@assert size(input, 1) == size(input, 2)	# Padded image!

	# Get the image origin to rotate around
        origin::Vector = ifloor(([size(input)...] .+ 1) ./ 2)

	# Allocate the output matrix
	output::Matrix{T} = Array(
		eltype(input),
		size(distances)...,
		size(angles)...
	)

	# Process all angles
	for a in angles
		# Rotate the image
		input_rotated::Matrix{T} = rotate(input, origin, a);

		for p in distances

		end	
	end

	return output
end
