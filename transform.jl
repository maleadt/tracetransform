include("functionals.jl")
include("geometry.jl")
include("sinogram.jl")
include("circus.jl")

using Images

function prepare_transform(input::Image{Float64}, angle_stepsize::Uint,
                           orthonormal::Bool)
    # Orthonormal P-functionals need a stretched image in order to ensure a
    # square sinogram
    if orthonormal
        ndiag = iceil(360/angle_stepsize)
        nsize::Uint = iceil(ndiag / sqrt(2))
        print_debug("Stretching input image to $(int(nsize)) squared.\n")
        input = resize(input, (nsize, nsize))
    end

    # Pad the image so we can freely rotate without losing information
    oldsize = size(input)
    input = pad(input)
    print_debug("Padded image from $(oldsize) to $(size(input)).\n")

    return input
end

function get_transform(input::Image{Float64},
    tfunctionals::Vector{TFunctionalWrapper},
    pfunctionals::Vector{PFunctionalWrapper},
    angle_stepsize::Uint, orthonormal::Bool, write_data::Bool)
    # Process all T-functionals
    print_debug("Calculating sinograms for given T-functionals\n")
    sinograms::Vector{Image{Float64}} = getSinograms(input, angle_stepsize,
                                           tfunctionals)
    for t in 1:length(tfunctionals)
        tfunctional = tfunctionals[t]
        if write_data
            # Save the sinogram trace
            writecsv("trace_$(tfunctional.functional).csv", sinograms[t].data);

            if want_log(debug_l)
                # Save the sinogram image
                # FIXME: why does imwrite generate a transposed image?
                imwrite(mat2gray(sinograms[t]),
                    "trace_$(tfunctional.functional).ppm")
            end
        end

        # Orthonormal functionals require the nearest orthonormal sinogram
        if orthonormal
            print_debug("Orthonormalizing sinogram\n")
            (sinogram_center, sinograms[t]) = nearest_orthonormal_sinogram(sinograms[t])
            for pfunctional in pfunctionals
                if pfunctional.functional == Hermite
                    pfunctional.arguments.center = sinogram_center
                end
            end
        end

        # Process all P-functionals
        for pfunctional in pfunctionals
            # Calculate the circus function
            print_debug("Calculating circus function using P-functional ",
                repr(pfunctional.functional), "\n")
            circus::Vector = getCircusFunction(
                sinograms[t],
                pfunctional
                )

            # Normalize
            normalized = zscore(circus)

            if write_data
                # Save the circus trace
                writecsv("trace_$(tfunctional.functional)-$(pfunctional.functional).csv",
                    normalized);
            end
        end
    end
end
