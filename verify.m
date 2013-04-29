clear all;

% Configuration
imageFile = '../test/Cam1_V1.pgm';
TFunctionals = [1 2 3 4 5];
PFunctionals = [1 2 3];   % >= 4 --> Hermite

% Get absolute path to file
curDir = pwd;
[imageRelPath, imageName, imageExt] = fileparts(imageFile);
cd(imageRelPath)
imagePath = pwd;
imageFile = strcat(imagePath, filesep, imageName, imageExt);
cd(curDir)

% Clean environment
curPath = getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH',getenv('PATH'));

% Compile C++ implementation
fprintf(1, 'Compiling C++ implementation...\n');
cd('cpp');
delete('transform');
[status, output] = system('make transform -j9');
cd(curDir);
if status > 0
    error('Could not compile C++ implementation');
    fprintf(1, output);
end

% C++ implementation
fprintf(1, 'Executing C++ implementation...\n');
delete('circus.dat');
TFunctionalsCell = cellstr(num2str(TFunctionals(:)));
TFunctionalsCell = strcat('-T ', TFunctionalsCell);
HermitePFunctionals = PFunctionals >= 4;
PFunctionalsSubstracted = PFunctionals;
PFunctionalsSubstracted(HermitePFunctionals) = PFunctionalsSubstracted(HermitePFunctionals)-3;
PFunctionalsCell = cellstr(num2str(PFunctionalsSubstracted(:)));
PFunctionalsCell(~HermitePFunctionals) = strcat('-P ', PFunctionalsCell(~HermitePFunctionals));
PFunctionalsCell(HermitePFunctionals) = strcat('-P H ', PFunctionalsCell(HermitePFunctionals));
tic
[status, output] = system(sprintf('./cpp/transform %s %s "%s"', strjoin(TFunctionalsCell, ' '), strjoin(PFunctionalsCell, ' '), imageFile));
toc
if status > 0
    error('Could not execute C++ implementation');
    fprintf(1, output);
end
Circus_2 = load('circus.dat');

% Restore environment
setenv('LD_LIBRARY_PATH',curPath);

% MATLAB implementation
fprintf(1, 'Executing MATLAB implementation...\n');
image = mat2gray(imread(imageFile));
tic
[Sinogram_1 Circus_1] = OrthTraceSign(image,TFunctionals,PFunctionals, 1, 1);
toc

% Compare circus functions
CircusDiff = sqrt(sum((Circus_2 - Circus_1).^2)./size(Circus_1,1)) ./ (max(Circus_1) - min(Circus_1));
T = 1;
P = 1;
figure
for i = 1:length(CircusDiff)
    fprintf(1, 'Difference using T-functional %s and P-functional %s: %.2f%%.\n', TFunctionalsCell{T}, PFunctionalsCell{P}, CircusDiff(i)*100);
    subplot(length(TFunctionals), length(PFunctionals), i);
    plot_1 = plot(Circus_1(:, i), 'b');
    hold all
    plot_2 = plot(Circus_2(:, i), 'r');
    title(sprintf('T-functional %s and P-functional %s', TFunctionalsCell{T}, PFunctionalsCell{P}))
    legend([plot_1 plot_2], 'MATLAB', sprintf('C++ (error=%.2f%%)', CircusDiff(i)*100))
    P = P+1;
    if P > length(PFunctionals)
       P = 1;
       T = T+1;
    end
end
