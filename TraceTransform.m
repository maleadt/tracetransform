function sinograms = TraceTransform(padded, tfunctionals, pfunctionals, angle_stepsize, orthonormal)
    sinograms = zeros(size(padded, 2), ceil(360/angle_stepsize), length(tfunctionals));

    angle_index = 0;
    for angle = [0:angle_stepsize:359]
        angle_index = angle_index + 1;
        rotated = imrotate(padded, angle, 'crop');
        for col = 1:size(rotated, 2)
            data = rotated(:, col);
            tfunctional_index = 0;
            for tfunctional = tfunctionals
                tfunctional_index = tfunctional_index + 1;
                switch tfunctional
                case 0
                    output = t_radon(data);
                case 1
                    output = t_1(data);
                otherwise
                    error('unknown T-functional')
                end
                sinograms(col, angle_index, tfunctional_index) = output;
            end
        end
    end
end

function median = find_weighted_median(data)
    total = sum(data);

    integral = 0;
    median = length(data);
    for i = 1:length(data)
        integral = integral + data(i);
        if 2*integral >= total
            median = i;
            break;
        end
    end
end

function output = t_radon(data)
    output = sum(data);
end

function integral = t_1(data)
     median = find_weighted_median(data);

    integral = 0;
    for r = median:length(data)
        integral = integral + data(r) * (r-median);
    end
end
