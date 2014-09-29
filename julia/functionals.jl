require("auxiliary")

#
# Auxiliary
#

function find_weighted_median(data)
    total = sum(data)

    integral = zero(eltype(data))
    for i in 1:length(data)
        @inbounds integral += data[i]
        if 2*integral >= total
            return i
        end
    end

    return length(data)
end

function hermite_polynomial(order::Uint, x::Number)
    if order == 0
        return 1;
    elseif order == 1
        return 2x;
    else
        return 2x*hermite_polynomial(order-1, x) -
        2*(order-1)*hermite_polynomial(order-2, x);
    end
end

function hermite_function(order::Uint, x::Number)
    return hermite_polynomial(order, x) / (
        sqrt(2^order * factorial(order) * sqrt(pi))
        * exp(x*x/2)
        )
end

#
# T-functionals
#

function t_radon(data)
    s = zero(eltype(data))
    for i=1:length(data)
        @inbounds s += data[i]
    end
    s
end

function t_1(data)
    median = find_weighted_median(data)

    integral = zero(eltype(data))
    for r in median:length(data)
        @inbounds integral += data[r] * (r-median)
    end
    return integral
end

function t_2(data)
    median = find_weighted_median(data)

    integral = zero(eltype(data))
    for r in median:length(data)
        @inbounds integral += data[r] * (r-median)*(r-median)
    end
    return integral
end

function t_3_prepare(datatype, rows, cols)
    precalc = Array(Complex{datatype}, rows)
    for r in 1:rows
        @inbounds precalc[r] = r * exp(5im*log(r))
    end
    return precalc
end

function t_4_prepare(datatype, rows, cols)
    precalc = Array(Complex{datatype}, rows)
    for r in 1:rows
        @inbounds precalc[r] = exp(3im*log(r))
    end
    return precalc
end

function t_5_prepare(datatype, rows, cols)
    precalc = Array(Complex{datatype}, rows)
    for r in 1:rows
        @inbounds precalc[r] = sqrt(r) * exp(4im*log(r))
    end
    return precalc
end

function t_345(data, precalc)
    squaredmedian = find_weighted_median(view(sqrt(data)))

    integral = zero(Complex{eltype(data)})
    factor = zero(Complex{eltype(data)}) + 5im
    for r in 1:(length(data)-squaredmedian)
        # From +1, since exp(i*log(0)) == 0
        @inbounds integral += precalc[r] * data[r+squaredmedian]
    end
    return abs(integral)
end

function t_6(data)
    median = find_weighted_median(view(sqrt(data)))
    positive = view(data, median:length(data))
    weighted = similar(positive)
    for i in 1:length(weighted)
        @inbounds weighted[i] = positive[i] * (i-1)
    end
    indices = sortperm(weighted)
    permuted = similar(positive)
    for i in 1:length(permuted)
        @inbounds permuted[i] = positive[indices[i]]
    end
    median = find_weighted_median(view(sqrt(permuted)))
    return weighted[indices[median]]
end

function t_7(data)
    median = find_weighted_median(data)
    positive = view(data, median:length(data))
    sorted = sort(positive)
    median = find_weighted_median(view(sqrt(sorted)))
    return sorted[median]
end


#
# P-functionals
#

function p_1(data)
    return mean(abs(diff(data)))
end

function p_2(data)
    sorted = sort(data)
    median = find_weighted_median(view(sorted))
    return sorted[median]
end

function p_3(data)
    return trapz(linspace(-1,1, length(data)), abs(fft(data)/length(data)).^4)
end

function p_hermite(data, order::Uint, center::Uint)
    # Discretize the [-10, 10] domain to fit the column iterator
    z = -10
    stepsize_lower = 10 / center
    stepsize_upper = 10 / (length(data) - 1 - center)

    # Calculate the integral
    integral = zero(eltype(data))
    for p in 1:length(data)
        @inbounds integral += data[p] * hermite_function(order, z)
        if z < 0
            z += stepsize_lower
        else
            z += stepsize_upper
        end
    end
    return integral
end
