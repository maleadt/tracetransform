function [Sinogram CircusF] = OrthTraceSign(I,Code_Tfunct,Code_Pfunct, angle_intrvl,flag)
        Ipadd = impaddingf(I,angle_intrvl, flag);     %% Padding the input image
        Sinogram = TraceTransform(Ipadd,Code_Tfunct,Code_Pfunct,angle_intrvl,flag); 
        CircusF = Apply_Pfunct(Sinogram, Code_Pfunct, flag);
        CircusF = zscore(CircusF);
end

function Ipadd = impaddingf(I,angle_intrvl, flag)
        if flag
                %% Resizing and padding the image with zeros
                ndiag = ceil(360/angle_intrvl);
                nrows = ceil(ndiag/sqrt(2));
                %% Resizing the image to generate a square sinogram
                Irsz = imresize(im2uint8(I), [nrows nrows]);
                M = nrows;
                N = nrows;
        else
                Irsz = im2uint8(I);
                M = size(Irsz,1);
                N = size(Irsz,2);

        end
        origin = floor(([M N]+1)/2);
         
        temp1 = M - 1 - origin(1);
        temp2 = N - 1 - origin(2);
        rLast = ceil(sqrt((temp1*temp1+temp2*temp2))) + 1;
        rFirst = -rLast ;
        Nbins = rLast - rFirst + 1;
        
        %% Padding the image 
        Ipadd = zeros(Nbins);
        originPadd = floor((size(Ipadd)+1)/2);
        df = originPadd - origin;
        Ipadd(1+df(1):M+df(1),1+df(2):N+df(2)) = Irsz; 
        Ipadd = im2double(uint8(Ipadd));
end