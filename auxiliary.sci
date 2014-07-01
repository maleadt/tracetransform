function out=zscore(in)
    _mean = mean(in)
    _std = stdev(in)

    out = zeros(in)
    for i = 1:length(in)
        out(i) = (in(i) - _mean) / _std
    end

    return output
endfunction

function out=bsxfun_ge(A, B)
    out = zeros(A)
    s = size(A, 2)
    for i = 1:size(A, 2) // col
        for j = 1:size(A, 1) // row
            if A(j, i) >= B(i)
                out(j, i) = 1
            else
                out(j, i) = 0
            end
        end
    end
endfunction
