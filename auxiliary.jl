
datawrite(file::String, data::Array) = datawrite(file, data, [])
function datawrite(filename::String, data::Array, headers::Vector)
        @assert length(headers) == 0 || length(headers) == size(data, 2)

        # Calculate column width
        rows = size(data, 1)
        cols = size(data, 2)
        widths::Vector = zeros(Uint, cols)
        for col in 1:cols
                if length(headers) > 0
                        widths[col] = length(headers[col])
                end
                for row in 1:rows
                        widths[col] = max(widths[col], length(string(data[row, col])))
                end
                widths[col] += 2        # add spacing
        end

        open(filename, "w") do file
                # Print headers
                if length(headers) > 0
                        write(file, "%  ");
                        for col in 1:cols
                                write(file, headers[col])
                                write(file, " "^(widths[col]-length(headers[col])))
                        end
                        write(file, "\n")
                end

                # Print data
                for row in 1:rows
                        write(file, "   ");
                        for col in 1:cols
                                write(file, string(data[row, col]))
                                write(file, " "^(widths[col]-length(string(data[row, col]))))
                        end
                        write(file, "\n")
                end
        end
end

function dataread(filename::String)
        data = Array(Number, 0, 0)

        file = open(filename) do file
                for line in each_line(file)
                        numbers::Vector = split(line)
                        if size(data, 2) == 0
                                data = Array(Number, 0, length(numbers))
                        elseif size(data, 2) != length(numbers)
                                error("inequally-sized rows")
                        end
                        data = vcat(data, numbers')
                end
        end

        return data
end

function zscore(input::Vector)
        local _mean = mean(input)
        local _std = std(input)

        output::Vector = similar(input)
        for i in 1:length(input)
                output[i] = (input[i] - _mean) / _std
        end

        return output
end

function trapz(x::Vector, y::Vector)
        @assert length(x) == length(y)

        sum = 0
        for i = 1:length(x)-1
                sum += (x[i+1] - x[i]) * (y[i+1] + y[i])
        end
        return sum * 0.5
end

mat2gray(input::Matrix) = uint8(clamp(input * 255 / max(input), 0, 255))