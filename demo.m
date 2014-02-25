transform

% Manage caller directory
addpath(pwd())
cd(directory);

% Check orthonormal
orthonormal_count = 0;
for i=1:length(pfunctionals)
    if pfunctionals(i) > 3
        orthonormal_count = orthonormal_count + 1;
    end
end
if orthonormal_count == 0
    orthonormal = false;
elseif orthonormal_count == length(pfunctionals)
    orthonormal = true;
else
    error('cannot mix orthonormal with non-orthonormal pfunctionals')
end

% Gather input data
image = mat2gray(imread(imageFile, 'pgm'));
[padded basename] = prepare_transform(image, imageFile, angle_interval, orthonormal);

if strcmp(program_mode, 'calculate')
    % Get output data
    [sinogram circus] = get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);

    % Save sinograms
    for t_i = 1:length(tfunctionals)
        t = tfunctionals(t_i);

        trace = sinogram(:, :, t_i);
        csvwrite(sprintf('%s-T%d.csv', basename, t), trace);
        % TODO: debug flag
        imwrite(mat2gray(trace), sprintf('%s-T%d.pgm', basename, t));
    end

    % Save circus functions
    for t_i = 1:length(tfunctionals)
        t = tfunctionals(t_i);
        for p_i = 1:length(pfunctionals)
            p = pfunctionals(p_i);
            if p >= 4
                p_real = p - 3;
                type = 'H';
            else
                p_real = p;
                type = 'P';
            end

            trace = circus(:, p_i + (t_i-1)*length(pfunctionals));
            csvwrite(sprintf('%s-T%d_%s%d.csv', basename, t, type, p_real), trace);
        end
    end

elseif strcmp(program_mode, 'benchmark')
    % Warm-up
    for i=1:3
        get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);
    end

    for i=1:iterations
       tstart = tic;
       get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);
       telapsed = toc(tstart);
       fprintf(1, 't_%g=%g\n', i, telapsed)
    end

elseif strcmp(program_mode, 'profile')
    % Warm-up
    get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);

    profile on
    get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);
    profile report
    uiwait

else
    error('invalid execution mode')
end
