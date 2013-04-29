clear all;

% Configuration
imageFile = '../test/Cam1_V1.pgm';
TFunctionals = [1 2 3 4 5];
PFunctionals = [1 2 3];   % >= 4 --> Hermite
REPS = 10;

% Benchmark
image = mat2gray(imread(imageFile));
minTime = Inf;
telapsed = zeros(1, REPS);
tic;
for i=1:REPS
   tstart = tic;
   benchmark_single
   telapsed(i) = toc(tstart);
   minTime = min(telapsed,minTime);
end
fprintf(1, '%g +/- %g\n', mean(telapsed), std(telapsed))

