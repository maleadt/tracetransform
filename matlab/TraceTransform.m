function [sinograms circus_functions] = TraceTransform(padded, tfunctionals, pfunctionals, angle_stepsize, orthonormal)
size_tfunct = length(tfunctionals);
size_pfunct = length(pfunctionals);
size_angles = ceil(360/angle_stepsize);
size_pcoord = size(padded, 2);
sinograms = zeros(size_pcoord, size_angles, size_tfunct);
circus_functions = zeros(size_angles,size_pfunct,size_tfunct);
    
%% Generating sinograms with T-functionals
    angle_index = 0;
    for angle = (0:angle_stepsize:359)
        angle_index = angle_index + 1;
        data = imrotate(padded, angle, 'crop');
            tfunctional_index = 0;
            for tfunctional = tfunctionals
                tfunctional_index = tfunctional_index + 1;
                switch tfunctional
                case 0
                    output = t_radon(data);
                case 1
                    output = t_1(data);
                case 2
                    output = t_2(data);
                case 3
                    output = t_3(data);
                case 4
                    output = t_4(data);
                case 5
                    output = t_5(data);
                case 6
                    output = t_6(data);
                case 7
                    output = t_7(data);
                otherwise
                    error('unknown T-functional')
                end
                sinograms(:, angle_index, tfunctional_index) = output;
            end
    end
    
%% Applying P-functionals
    pfunctional_index = 0;
    for pfunctional = pfunctionals 
        pfunctional_index = pfunctional_index + 1;
        switch pfunctional
            case 1
                signatures = p_1(sinograms);
            case 2
                signatures = p_2(sinograms, size_pcoord, size_angles, size_tfunct);  
            case 3
                signatures = p_3(sinograms, size_pcoord);
            otherwise
                error('unknown P-functional')
        end
        circus_functions(:,pfunctional_index,:) = signatures;
    end
   
    circus_functions = zscore(reshape(circus_functions,size_angles,size_pfunct*size_tfunct));
    
end

%% Auxiliary functions
function [R loc] = find_R(data)
    Accum = cumsum(data);
      loc = bsxfun(@ge,2*Accum,Accum(end,:));
  R_shift = cumsum(loc);
        R = [zeros(1,size(data,2));R_shift(1:end-1,:)];
end

function [R1 loc] = find_R1(data)
    Accum = cumsum(data.^(1/2));
      loc = bsxfun(@ge,2*Accum,Accum(end,:));
 R1_shift = cumsum(loc);
       R1 = [zeros(1,size(data,2));R1_shift(1:end-1,:)];
end

%% T-functionals
% Radon transform
function output = t_radon(data)
    output = sum(data);
end

% T1 functional
function integral = t_1(data)
           R = find_R(data);
    integral = sum( R .* data );
end

% T2 functional
function integral = t_2(data)
           R = find_R(data);
    integral = sum( (R.^2) .* data );
end

% T3 functional
function integral = t_3(data)
                Weight = zeros(size(data));
                    R1 = find_R1(data);
                 valid = R1 ~= 0;
         valid_weights = exp(5 *1i* log(R1(valid)));
         Weight(valid) = valid_weights;
              integral = abs(sum(Weight .* R1 .* data));   
end

% T4 functional
function integral = t_4(data)
                Weight = zeros(size(data));
                    R1 = find_R1(data);
                 valid = R1 ~= 0;
         valid_weights = exp(3 *1i* log(R1(valid)));
         Weight(valid) = valid_weights;
              integral = abs(sum(Weight .* data));
end

% T5 functional
function integral = t_5(data)
                Weight = zeros(size(data));
                    R1 = find_R1(data);
                 valid = R1 ~= 0;
         valid_weights = exp(4 *1i* log(R1(valid)));
         Weight(valid) = valid_weights;
              integral = abs(sum(Weight .* (R1.^(1/2)) .* data));
end


% T6 functional: median{r1*f(r1),(f(r1))^(1/2)}
function weighted_median = t_6(data)
   [nx,ny] = size(data);
    
   % Getting the domain of R1
    [R1 valid] = find_R1(data);
     
   % Masking the input data to get samples where r1 > 0
     F = data .* valid;
     
   % Getting r1*f(r1) and sorting 
    [F_R1_sort idx] = sort(F .* R1);
    
   % Permuting f(r1) accordingly 
     index = sub2ind([nx ny],idx,repmat(1:ny,nx,1));
     F_permuted = F(index);
     
   % Finding the positions of the weighted median
     data_accum = cumsum(F_permuted.^(1/2));
           mask = bsxfun(@ge,2*data_accum,data_accum(end,:));  
     [~,median_indices] = max(mask);
   
   % Retrieve the weighted median for each column
     index = sub2ind([nx ny],median_indices,1:ny);
     weighted_median = F_R1_sort(index);
end

% T7 functional: median{f(r),(f(r))^(1/2)}
function weighted_median = t_7(data)
   [nx,ny] = size(data);

   % Getting the domain of R1
     [~,valid] = find_R(data);
     
   % Masking and sorting the data for r > 0
     F = sort(data .* valid);
     
   % Finding the location of the weighted median
      data_accum = cumsum(F.^(1/2));
            mask = bsxfun(@ge,2*data_accum,data_accum(end,:));  
     [~,median_indices] = max(mask);
   
   % Retrieve the weighted median for each column
     index = sub2ind([nx ny],median_indices,1:ny);
     weighted_median = F(index);
end

%% P-functionals
% P1-functional
function signatures = p_1(sinogram)
signatures = squeeze(sum(abs(diff(sinogram))));
end

% P2-functional
function signatures = p_2(sinogram,nx,ny,nz)
     
   % Sorting input datat
          F = sort(sinogram);
     
   % Finding the location of the weighted median
      data_accum = cumsum(F);
            mask = bsxfun(@ge,2*data_accum,data_accum(end,:,:));  
     [~,median_indices] = max(mask);
   
   % Retrieve the weighted median for each column
           if nz > 1
             index = sub2ind([nx ny nz],squeeze(median_indices),repmat((1:ny)',1,nz),repmat((1:nz),ny,1));
           else
             index = sub2ind([nx ny],median_indices,1:ny);
           end
        signatures = F(index);
end

% P3-functional
function signatures = p_3(sinogram,L)
signatures = squeeze(trapz(linspace(-1,1,L),abs(fft(sinogram)/L).^4));
end
