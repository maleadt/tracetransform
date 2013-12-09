#
# Auxiliary
#

function find_weighted_median(data::StridedVector)
    total = sum(data)

    integral = 0
    for i in 1:length(data)
        integral += data[i]
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

function t_radon(data::StridedVector)
    return sum(data)
end

function t_1(data::StridedVector)
    median = find_weighted_median(data)

    integral = 0
    for r in median:length(data)
        integral += data[r] * (r-median)
    end
    return integral
end

function t_2(data::StridedVector)
    median = find_weighted_median(data)

    integral = 0
    for r in median:length(data)
        integral += data[r] * (r-median)*(r-median)
    end
    return integral
end

function t_3(data::StridedVector)
    squaredmedian = find_weighted_median(sqrt(data))

    integral = 0 + 0im
    factor = 0 + 5im
    for r1 in squaredmedian+1:length(data)
        # From +1, since exp(i*log(0)) == 0
        integral += exp(factor*log(r1-squaredmedian)) *
        data[r1] * (r1-squaredmedian)
    end
    return abs(integral)
end

function t_4(data::StridedVector)
    squaredmedian = find_weighted_median(sqrt(data))

    integral = 0 + 0im
    factor = 0 + 3im
    for r1 in squaredmedian+1:length(data)
        # From +1, since exp(i*log(0)) == 0
        integral += exp(factor*log(r1-squaredmedian)) *
        data[r1]
    end
    return abs(integral)
end

function t_5(data::StridedVector)
    squaredmedian = find_weighted_median(sqrt(data))

    integral = 0 + 0im
    factor = 0 + 4im
    for r1 in squaredmedian+1:length(data)
        # From +1, since exp(i*log(0)) == 0
        integral += exp(factor*log(r1-squaredmedian)) *
        data[r1] * sqrt(r1-squaredmedian)
    end
    return abs(integral)
end


#
# P-functionals
#

# Julia issue #5074
import Base.diff
diff(a::SubArray) = diff(collect(a))

function p_1(data::StridedVector)
    return mean(abs(diff(data)))
end

function p_2(data::StridedVector)
    median = find_weighted_median(data)
    return data[median]
end

function p_3(data::StridedVector)
    return sum(abs(fft(data)).^4)
end

function p_hermite(data::StridedVector, order::Uint, center::Uint)
    # Discretize the [-10, 10] domain to fit the column iterator
    z = -10
    stepsize_lower = 10 / center
    stepsize_upper = 10 / (length(data) - 1 - center)

    # Calculate the integral
    integral = 0
    for p in 1:length(data)
        integral += data[p] * hermite_function(order, z)
        if z < 0
            z += stepsize_lower
        else
            z += stepsize_upper
        end
    end
    return integral
end
