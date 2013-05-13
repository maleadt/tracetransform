% Check arguments
if exist('imageFile') == 0
        error('imageFile not defined');
elseif exist('tfunctionals') == 0
        error('tfunctionals not defined');
elseif exist('pfunctionals') == 0
        error('pfunctionals not defined');
end

% Gather data
image = mat2gray(imread(imageFile, 'pgm'));
[sinogram circus] = OrthTraceSign(image, tfunctionals, pfunctionals, 1, 0); % TODO: flag

% Save sinograms
for t_i = 1:length(tfunctionals)
    t = tfunctionals(t_i);
    
    trace = sinogram(:, :, t_i);
    csvwrite(sprintf('trace_T%d.csv', t), trace);
    % TODO: debug flag
    imwrite(mat2gray(trace), sprintf('trace_T%d.pgm', t));
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
        csvwrite(sprintf('trace_T%d-%s%d.csv', t, type, p_real), trace);
    end
end
