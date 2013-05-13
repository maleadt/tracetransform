% Check arguments
if exist('imageFile') == 0
        error('imageFile not defined');
elseif exist('tfunctionals') == 0
        error('tfunctionals not defined');
elseif exist('pfunctionals') == 0
        error('pfunctionals not defined');
elseif exist('iterations') == 0
        error('iterations not defined');
end

% Benchmark
image = mat2gray(imread(imageFile, 'pgm'));
telapsed = zeros(1, iterations);
tic;
for i=1:iterations
   tstart = tic;
   OrthTraceSign(image, tfunctionals, pfunctionals, 1, 0); % TODO: flag
   telapsed(i) = toc(tstart);
end
fprintf(1, 'Total execution time for %d iterations: %.2f ms.\n', iterations, 1000*sum(telapsed))
fprintf(1, 'Average execution time: %.2f +/- %.2f ms.\n', 1000*mean(telapsed), 1000*std(telapsed))