require("functionals")
require("enum")

@enum TFunctional NoT Radon T1 T2 T3 T4 T5

type TFunctionalArguments
end

type TFunctionalWrapper
        functional::TFunctional
        arguments::TFunctionalArguments

        function TFunctionalWrapper()
                functional = NoT
                arguments = TFunctionalArguments()
                new(functional, arguments)
        end
end

function parse_tfunctionals(args)
        tfunctionals::Vector = TFunctionalWrapper[]
        for functional in args
                wrapper = TFunctionalWrapper()
                if functional == "0"
                        wrapper.functional = Radon
                elseif functional == "1"
                        wrapper.functional = T1
                elseif functional == "2"
                        wrapper.functional = T2
                elseif functional == "3"
                        wrapper.functional = T3
                elseif functional == "4"
                        wrapper.functional = T4
                elseif functional == "5"
                        wrapper.functional = T5
                else
                        error("unknown T-functional")
                end
                push!(tfunctionals, wrapper)
        end

        return tfunctionals
end

function getSinogram(
        input::Matrix,
        tfunctional::TFunctionalWrapper)
        @assert size(input, 1) == size(input, 2)        # Padded image!

        # Get the image origin to rotate around
        origin::Vector = ifloor(flipud([size(input)...] .+ 1) ./ 2)

        # Allocate the output matrix
        output::Matrix = Array(
                eltype(input),
                cols(input),    # rows
                360             # cols
                )

        # Process all angles
        for a in 0:359
                # Rotate the image
                input_rotated::Matrix = rotate(input, origin, a)

                # Process all projection bands
                for p in 1:cols(input)-1
                        if tfunctional.functional == Radon
                                output[p, a+1] = t_radon(vec(input_rotated[:, p]))
                        elseif tfunctional.functional == T1
                                output[p, a+1] = t_1(vec(input_rotated[:, p]))
                        elseif tfunctional.functional == T2
                                output[p, a+1] = t_2(vec(input_rotated[:, p]))
                        elseif tfunctional.functional == T3
                                output[p, a+1] = t_3(vec(input_rotated[:, p]))
                        elseif tfunctional.functional == T4
                                output[p, a+1] = t_4(vec(input_rotated[:, p]))
                        elseif tfunctional.functional == T5
                                output[p, a+1] = t_5(vec(input_rotated[:, p]))
                        end
                end
        end

        return output
end
