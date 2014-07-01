function x=interpolate(im0, p)
    // Get fractional and integral part of the coordinates
    i_int = int(p(1))
    i_fract = p(1) - i_int
    j_int = int(p(2))
    j_fract = p(2) - j_int

    // Bilinear interpolation
    x = (im0(i_int,   j_int)   * (1-j_fract) * (1-i_fract) + ...
         im0(i_int,   j_int+1) * j_fract     * (1-i_fract) + ...
         im0(i_int+1, j_int)   * (1-j_fract) * i_fract + ...
         im0(i_int+1, j_int+1) * j_fract     * i_fract)
endfunction

function im=imrotate(im0, origin, angle)
    // Calculate part of transform matrix
    angle_cos = cosd(angle)
    angle_sin = sind(angle)

    // Allocate output matrix
    im = zeros(im0)

    // Process all pixels
    s = size(im0)
    for i = 1:s(1)
        for j = 1:s(2)
            // Get the source pixel
            t = [i-origin(1), j-origin(2)]
            r = [-t(2)*angle_sin + t(1)*angle_cos + origin(1) ...
                  t(2)*angle_cos + t(1)*angle_sin + origin(2)]

            // Copy if within bounds
            if 1 <= r(1) & r(1) < s(1) & 1 <= r(2) & r(2) < s(2)
                im(i, j) = interpolate(im0, r)
            end
        end
    end
endfunction
