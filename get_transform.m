function [sinogram circus_function] = get_transform(padded ,tfunctionals, pfunctionals, angle_stepsize, orthonormal)
    [sinogram circus_function] = TraceTransform(padded, tfunctionals, pfunctionals, angle_stepsize, orthonormal);     
end
