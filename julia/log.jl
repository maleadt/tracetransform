@enum LogLevel trace_l debug_l info_l warning_l error_l fatal_l
THRESHOLD = info_l

function set_threshold(level::LogLevel)
    global THRESHOLD::LogLevel
    THRESHOLD::LogLevel = level
end

function want_log(level::LogLevel)
    return (level.n >= (THRESHOLD::LogLevel).n)
end

function print_checked(level::LogLevel, msg::String...)
    if want_log(level)
        print(msg...)
    end
end

print_fatal(msg::String...) = print_checked(fatal_l, msg...)
print_error(msg::String...) = print_checked(error_l, msg...)
print_warning(msg::String...) = print_checked(warning_l, msg...)
print_info(msg::String...) = print_checked(info_l, msg...)
print_debug(msg::String...) = print_checked(debug_l, msg...)
print_trace(msg::String...) = print_checked(trace_l, msg...)
