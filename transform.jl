require("functionals")
require("sinogram")
require("circus")

function prepare_transform(input, orthonormal)
        # Orthonormal P-functionals need a stretched image in order to ensure a
        # square sinogram
        if orthonormal
                ndiag = 360
                nsize = iceil(ndiag / sqrt(2))
                input = resize(input, nsize, nsize)
        end

        # Pad the image so we can freely rotate without losing information
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

function get_transform(input, tfunctionals, pfunctionals, orthonormal)
        # Process all T-functionals
        for tfunctional in tfunctionals
                # Calculate the trace transform sinogram
                println("Calculating $(tfunctional.functional) sinogram")
                const sinogram = getSinogram(
                        input,
                        tfunctional
                )

                if want_log(debug)
                        # Save the sinogram image
                        imwrite(share(input_image, mat2gray(sinogram)), "trace_$(tfunctional.functional).pbm")
                end

                # Save the sinogram data
                writecsv("trace_$(tfunctional.functional).csv", sinogram);

                # Orthonormal functionals require the nearest orthonormal sinogram
                if orthonormal
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
                        println("Calculating $(pfunctional.functional) circus function for $(tfunctional.functional) sinogram")
                        circus::Vector = getCircusFunction(
                                sinogram,
                                pfunctional
                        )

                        # Normalize
                        normalized = zscore(circus)

                        # Save the circus data
                        writecsv("trace_$(tfunctional.name)-$(pfunctional.name).csv", normalized);
                end
        end
end