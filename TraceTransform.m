function sinograms = TraceTransform(padded, tfunctionals, pfunctionals, angle_stepsize, orthonormal)
    sinograms = zeros(size(padded, 2), ceil(360/angle_stepsize), length(tfunctionals));

    angle_index = 0;
    for angle = [0:angle_stepsize:359]
        angle_index = angle_index + 1;
        rotated = imrotate(padded, angle, 'crop');
        %for col = 1:size(rotated, 2)
            data = rotated;%rotated(:, col);
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
                %sinograms(col, angle_index, tfunctional_index) = output;
                sinograms(:, angle_index, tfunctional_index) = output;
            end
        %end
    end
end

% function median_value = find_weighted_median(data)
%     total = sum(data);
% 
%     integral = 0;
%     median_value = length(data);
%     for i = 1:length(data)
%         integral = integral + data(i);
%         if 2*integral >= total
%             median_value = i;
%             break;
%         end
%     end

%% Simplifying code
%     integral = cumsum(data);                       
% median_value = find(2*integral >= integral(end),1);
% end

%function median_value = find_wsqrt_median(data)
%     total = sum(data);
% 
%     integral = 0;
%     median_value = length(data);
%     for i = 1:length(data)
%         integral = integral + data(i);
%         if 2*integral >= total
%             median_value = i;
%             break;
%         end
%     end

%% Simplifying code
%    integral = cumsum(data.^(1/2));                       
%median_value = find(2*integral >= integral(end),1);
%end

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

%% Radon transform
function output = t_radon(data)
    output = sum(data);
end

%% T1 functional
function integral = t_1(data)
%     median_value = find_weighted_median(data);   % Renaming median by median_value to avoid conflicts with Matlab's predefined function
% 
%     integral = 0;
%     for r = median_value:length(data)
%         integral = integral + data(r) * (r-median_value);
%     end  
           R = find_R(data);
    integral = sum( R .* data );
end

%% T2 functional
function integral = t_2(data)
%     median_value = find_weighted_median(data);   % Renaming median by median_value to avoid conflicts with Matlab's predefined function
% 
%     integral = 0;
%     for r = median_value:length(data)
%         integral = integral + data(r) * (r-median_value)^2;
%     end  
           R = find_R(data);
    integral = sum( (R.^2) .* data );
end

%% T3 functional
function integral = t_3(data)
                Weight = zeros(size(data));
                    R1 = find_R1(data);
                 valid = R1 ~= 0;
         valid_weights = exp(5 *1i* log(R1(valid)));
         Weight(valid) = valid_weights;
              integral = abs(sum(Weight .* R1 .* data));   
end

%% T4 functional
function integral = t_4(data)
                Weight = zeros(size(data));
                    R1 = find_R1(data);
                 valid = R1 ~= 0;
         valid_weights = exp(3 *1i* log(R1(valid)));
         Weight(valid) = valid_weights;
              integral = abs(sum(Weight .* data));
end

%% T5 functional
function integral = t_5(data)
                Weight = zeros(size(data));
                    R1 = find_R1(data);
                 valid = R1 ~= 0;
         valid_weights = exp(4 *1i* log(R1(valid)));
         Weight(valid) = valid_weights;
              integral = abs(sum(Weight .* (R1.^(1/2)) .* data));
end


%% T6 functional: median{r1*f(r1),(f(r1))^(1/2)}
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

%% T7 functional: median{f(r),(f(r))^(1/2)}
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
     weighted_median = data_masked(index);
end

% % T7 functional
% function integral = t_7(data)
%          integral = zeros(1,size(data,2));     
%     for i = 1:size(data,2)
%         vec_col = data(:,i);
%               c = find_weighted_median(vec_col);
%       data_sort = sort(vec_col(c:end));
%           index = find_wsqrt_median(data_sort);
%     integral(i) = data_sort(index);
%     end
% end
