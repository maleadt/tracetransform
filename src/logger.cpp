//
// Configuration
//


// Header include
#include "logger.hpp"

// Standard library
#include <stddef.h>                     // for size_t
#include <cassert>                      // for assert
#include <ctime>                        // for localtime, strftime, time, etc
#include <sstream>                      // for stringstream

// Null stream
std::ostream cnull(0);

// Logger instantiation
Logger logger;


//
// Construction and destruction
//
//

Logger::Logger()
                : _buf(std::cout.rdbuf())
{
        // Replace the default stream buffer
        // TODO: do this for cerr as well, and use it for warnings and errors
        std::cout.flush();
        std::cout.rdbuf(&_buf);

        // Default configuration
        settings.threshold = info;
        settings.prefix_timestamp = false;
        settings.prefix_level = false;
}


//
// Logging
//

// TODO: breaks when doing clog(...) << "foo \n bar"
std::ostream& Logger::log(LogLevel level)
{
        if (level <= settings.threshold) {
                // Manage prefixes
                char last_char = _buf.last_char();
                if (last_char == '\r' || last_char == '\n') {
                        if (settings.prefix_timestamp)
                                std::cout << timestamp() << "  ";
                        if (settings.prefix_level)
                                std::cout << prefix(level) << "\t";
                }

                return std::cout;
        } else {
                return cnull;
        }
}


//
// Auxiliary
//

std::string Logger::timestamp()
{
        time_t datetime;
        time(&datetime);

        std::string buffer;
        buffer.resize(32);

        size_t len = strftime(&buffer[0], buffer.length(),
                        "%Y-%m-%dT%H:%M:%S%z", localtime(&datetime));
        assert(len);
        buffer.resize(len);

        return buffer;
}

std::string Logger::prefix(LogLevel level)
{
        switch (level) {
                case fatal:
                        return "FATAL";
                case error:
                        return "ERROR";
                case warning:
                        return "WARNING";
                case info:
                        return "INFO";
                case debug:
                        return "DEBUG";
                case trace:
                        return "TRACE";
                default:
                        int base;
                        if (level < fatal)
                                base = fatal;
                        else
                                base = trace;
                        int diff = level - base;

                        std::stringstream ss;
                        ss << prefix((LogLevel) base) << std::showpos << diff;

                        return ss.str();

        }
}

//
// Syntax sugar
//

std::ostream& clog(LogLevel level)
{
        return logger.log(level);
}
