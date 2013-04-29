%clear all;

% Configuration
imageFile = '../test/Cam1_V1.pgm';
TFunctionals = [1 2 3 4 5];
PFunctionals = [4 5 6];   % >= 4 --> Hermite

% Gather data
image = mat2gray(imread(imageFile));
[Sinogram Circus] = OrthTraceSign(image, TFunctionals, PFunctionals, 1, 1);

% Save sinograms
for t = 1:length(TFunctionals)
    imwrite(mat2gray(Sinogram(:, :, t)), sprintf('trace_T%d.pgm', t));
    
    trace = Sinogram(:, :, t);
    save(sprintf('trace_T%d.dat', t), '-ascii', 'trace');
end

% Save circus functions
for t = 1:length(TFunctionals)
    for p = 1:length(PFunctionals)
        if p >= 4
            p_real = p - 3;
            type = 'H';
        else
            p_real = p;
            type = 'P';
        end
        
        trace = Circus(:, p + (t-1)*length(PFunctionals));
        save(sprintf('trace_T%d-%s%d.dat', t, type, p_real), '-ascii', 'trace');
    end
end
