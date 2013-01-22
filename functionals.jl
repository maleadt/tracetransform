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
        for r in 1:length(data)-median
                integral += data[r+median] * r
        end
        return integral
end


#
# P-functionals
#

function p_1(data::Vector)
        return mean(abs(diff(data)))
end
