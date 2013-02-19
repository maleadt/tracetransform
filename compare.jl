require("auxiliary")

using ArgParse

function main(args)
        #
        # Initialization
        #
        
        # Parse the command arguments
        s = ArgParseSettings("Allowed options")
        @add_arg_table s begin
                "--sinogram"
                        action = :store_true
                        help = "compare sinograms"
                "--circus"
                        action = :store_true
                        help = "compare circus functions"
                "--t-functional", "-T"
                        action = :append_arg
                        help = "T-functionals"
                        # TODO; required = true
                "--p-functional", "-P"
                        action = :append_arg
                        help = "P-functionals"
                "input"
                        help = "folder with generated images and data files"
                        required = true
                "reference"
                        help = "folder with reference images and data files"
                        required = true
        end
        parsed_args = parse_args(args, s)


        #
        # Sinogram comparison
        #
        
        if parsed_args["sinogram"]
                # TODO
        end


        #
        # Circus function comparison
        #
        
        if parsed_args["circus"]
                for t in parsed_args["t-functional"]
                        t_real = "T$(t)"
                        for p in parsed_args["p-functional"]
                                if p[1] == 'H'
                                        p_real = p
                                else
                                        p_real = "P$(p)"
                                end

                                fn = "trace_$(t_real)-$(p_real).dat"
                                input_fn = string(parsed_args["input"], '/', fn)
                                reference_fn = string(parsed_args["reference"], '/', fn)
                                if !isfile(input_fn)
                                        error("Input datafile for $(t_real) $(p_real) doesn't exist")
                                elseif !isfile(input_fn)
                                        error("Reference datafile for $(t_real) $(p_real) doesn't exist")
                                end
                                
                                input = vec(dataread(input_fn))
                                input = map((str) -> parse_float(str), input)

                                reference = vec(dataread(reference_fn))
                                reference = map((str) -> parse_float(str), reference)

                                @assert length(input) == length(reference)

                                nmrse = sqrt(sum((input - reference).^2)./length(input)) ./ (max(reference) - min(reference))
                                
                                println("$(t_real)-$(p_real): $(round(100*nmrse, 2))% error")
                        end
                end
        end
end

main(ARGS)
