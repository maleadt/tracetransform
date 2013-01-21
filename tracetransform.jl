require("functionals.jl")

function getTraceTransform(
	input::Matrix,
	angles::Vector,
	distances::Vector)
	@assert size(input, 1) == size(input, 2)	# Padded image!

	# Get the image origin to rotate around
        origin::Vector = ifloor(([size(input)...] .+ 1) ./ 2)

	# Allocate the output matrix
	output::Matrix = Array(
		eltype(input),
		size(distances)...,	# rows
		size(angles)...		# cols
	)

	# Process all angles
	a_i = 1
	for a in angles
		println("Angle ", a)

		# Rotate the image
		input_rotated::Matrix = rotate(input, origin, a)

		# Process all projection bands
		p_i = 1
		for p in distances
			output[p_i, a_i] = t_radon(squeeze(input_rotated[p, :]))
			p_i += 1
		end	
		a_i += 1
	end

	return output
end
