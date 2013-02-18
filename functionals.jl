#
# Wrappers
#

abstract Functional

function call(f::Functional, data::Vector)
        return f.functional(data)
end

type SimpleFunctional <: Functional
        functional::Function
        name::String
end

type HermiteFunctional <: Functional
        name::String
        order::Uint
        center::Uint
end

function call(f::HermiteFunctional, data::Vector)
        return p_hermite(datam, order, center)
end


#
# Auxiliary
#

function find_weighted_median(data::Vector)
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

#
# T-functionals
#

function t_radon(data::Vector)
        return sum(data)
end

function t_1(data::Vector)
        median = find_weighted_median(data)

        integral = 0
        for r in median:length(data)
                integral += data[r] * (r-median)
        end
        return integral
end

function t_2(data::Vector)
        median = find_weighted_median(data)

        integral = 0
        for r in median:length(data)
                integral += data[r] * (r-median)*(r-median)
        end
        return integral
end

function t_3(data::Vector)
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

function t_4(data::Vector)
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

function t_5(data::Vector)
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

function p_1(data::Vector)
        return mean(abs(diff(data)))
end

function p_2(data::Vector)
        median = find_weighted_median(data)
        return data[median]
end

function p_3(data::Vector)
        return sum(abs(fft(data)).^4)
end
