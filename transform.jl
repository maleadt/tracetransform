require("functionals")
require("sinogram")
require("circus")

function prepare_transform(input, orthonormal)
        # Orthonormal P-functionals need a stretched image in order to ensure a
        # square sinogram
        if orthonormal
                ndiag = 360
                nsize = iceil(ndiag / sqrt(2))
                print_debug("Stretching input image to $nsize squared.\n")
                input = resize(input, nsize, nsize)
        end

        # Pad the image so we can freely rotate without losing information
        input = pad(input)
        print_debug("Padded image to $(rows(input))x$(cols(input)).\n")

        return input
end

function get_transform(input, tfunctionals, pfunctionals, orthonormal, write_data::Bool)
        # Process all T-functionals
        for tfunctional in tfunctionals
                # Calculate the trace transform sinogram
                print_debug("Calculating sinogram using T-functional $(tfunctional.functional)\n")
                const sinogram = getSinogram(
                        input,
                        tfunctional
                )

                if write_data
                        # Save the sinogram trace
                        writecsv("trace_$(tfunctional.functional).csv", sinogram);

                        if want_log(debug_l)
                                # Save the sinogram image
                                imwrite(share(input_image, mat2gray(sinogram)),
                                        "trace_$(tfunctional.functional).pbm")
                        end
                end

                # Orthonormal functionals require the nearest orthonormal sinogram
                if orthonormal
                        print_debug("Orthonormalizing sinogram\n")
                        (sinogram_center, sinogram) = nearest_orthonormal_sinogram(sinogram)
                        for pfunctional in pfunctionals
                                if pfunctional.functional == Hermite
                                        pfunctional.arguments.center = sinogram_center
                                end
                        end
                end

                # Process all P-functionals
                for pfunctional in pfunctionals
                        # Calculate the circus function
                        print_debug("Calculating circus function using P-functional $(pfunctional.functional)\n")
                        circus::Vector = getCircusFunction(
                                sinogram,
                                pfunctional
                        )

                        # Normalize
                        normalized = zscore(circus)

                        if write_data
                                # Save the circus trace
                                writecsv("trace_$(tfunctional.name)-$(pfunctional.name).csv",
                                        normalized);
                        end
                end
        end
end