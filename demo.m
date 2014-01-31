transform

% Check arguments
if exist('imageFile') == 0
    error('imageFile not defined');
elseif exist('tfunctionals') == 0
    error('tfunctionals not defined');
elseif exist('pfunctionals') == 0
    error('pfunctionals not defined');
elseif exist('program_mode') == 0
    error('mode not defined');
    if strcmp(program_mode, 'benchmark')
        if exist('iterations') == 0
            error('iterations not defined');
        end
    end
end

% Check optional arguments
if exist('angle_interval') == 0
    angle_interval = 1;
end
if exist('directory') ~= 0
    addpath(pwd())
    cd(directory);
end

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
    get_transform(padded, tfunctionals, pfunctionals, angle_interval, orthonormal);

    for i=1:iterations
       tstart = tic;
       get_transform(padded, tfunctionals, pfunctionals, 1, orthonormal);
       telapsed = toc(tstart);
       fprintf(1, 't_%g=%g\n', i, telapsed)
    end

else
    error('invalid execution mode')
end
