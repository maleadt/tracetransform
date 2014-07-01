function rottest(filename, angle)
    // Read image
    image = im2double(imread(filename, 'pgm'));
    printmat(255*image)

    // Rotate image
    image_rotated = imrotate(image, str2num(angle), 'bilinear', 'crop');

    // Output image
    fprintf(1, '\n');
    output = round(255*mat2gray(image_rotated));
    printmat(output)
endfunction

function printmat(input)
    for row=1:size(input, 1)
        for col=1:size(input, 2)
            pixel = input(row, col);
            if pixel == 0
                decimals = 1;
            else
                decimals = floor(log10(pixel))+1;
            end
            fprintf(1, '%d%s', pixel, repmat(' ',1,4-decimals));
        end
        fprintf(1, '\n');
    end
endfunction
