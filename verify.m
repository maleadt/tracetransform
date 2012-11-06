clear all;

% Configuration
imageFile = '../test/Cam1_V1.pgm';
TFunctionals = [1 2 3 4 5];
PFunctionals = [4 5 6];   % >= 4 --> Hermite

% Get absolute path to file
curDir = pwd;
[imageRelPath, imageName, imageExt] = fileparts(imageFile);
cd(imageRelPath)
imagePath = pwd;
imageFile = strcat(imagePath, filesep, imageName, imageExt);
cd(curDir)

% MATLAB algorithm
fprintf(1, 'Executing MATLAB implementation...\n');
image = mat2gray(imread(imageFile));
[Sinogram_1 Circus_1] = OrthTraceSign(image,TFunctionals,PFunctionals, 1, 1);

% Clean environment
curPath = getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH',getenv('PATH'));

% Compile C++ algorithm
fprintf(1, 'Compiling C++ implementation...\n');
cd('../reference/build_release');
delete('transform');
[status, output] = system('make transform -j9');
cd(curDir);

% C++ algorithm
fprintf(1, 'Executing C++ implementation...\n');
TFunctionalsCell = cellstr(num2str(TFunctionals(:)));
HermitePFunctionals = PFunctionals >= 4;
PFunctionals(HermitePFunctionals) = PFunctionals(HermitePFunctionals)-3;
PFunctionalsCell = cellstr(num2str(PFunctionals(:)));
PFunctionalsCell(HermitePFunctionals) = strcat('H', PFunctionalsCell(HermitePFunctionals));
[status, output] = system(sprintf('../reference/build_release/transform "%s" %s %s', imageFile, strjoin(TFunctionalsCell, ','), strjoin(PFunctionalsCell, ',')));
Circus_2 = load('circus.dat');

% Restore environment
setenv('LD_LIBRARY_PATH',curPath);

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