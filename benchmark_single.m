% Configuration
imageFile = '../test/Cam1_V1.pgm';
TFunctionals = [1 2 3 4 5];
PFunctionals = [1 2 3];   % >= 4 --> Hermite

image = mat2gray(imread(imageFile));
[Sinogram_1 Circus_1] = OrthTraceSign(image,TFunctionals,PFunctionals, 1, 1);
