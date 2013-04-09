//
// Configuration
//


// Header include
#include "logger.hpp"

// Standard library
#include <sstream>
#include <ctime>
#include <cassert>
#include <iomanip>

// Null stream
std::ostream cnull(0);

// Logger instantiation
Logger logger;


//
// Construction and destruction
//
//
Logger::Logger() : _buf(std::cout.rdbuf()) {
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

	size_t len = strftime(&buffer[0], buffer.length(), "%Y-%m-%dT%H:%M:%S%z", localtime(&datetime));
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


//
// Auxiliary functions
//

std::string hexdump(void* x, unsigned long len, unsigned int w)
{
  std::ostringstream osDump;
  std::ostringstream osNums;
  std::ostringstream osChars;
  std::string szPrevNums;
  bool bRepeated = false;
  unsigned long i;

  for(i = 0; i <= len; i++)
  {
     if(i < len)
     {
        char c = (char)*((char*)x + i);
        unsigned int n = (unsigned int)*((unsigned char*)x + i);
        osNums << std::setbase(16) << std::setw(2) << std::setfill('0') << n << " ";
        if(((i % w) != w - 1) && ((i % w) % 8 == 7))
           osNums << "- ";
        osChars << (iscntrl(c) ? '.' : c);
     }

     if(osNums.str().compare(szPrevNums) == 0)
     {
        bRepeated = true;
        osNums.str("");
        osChars.str("");
        if(i == len - 1)
           osDump << "*" << std::endl;
        continue;
     }

     if(((i % w) == w - 1) || ((i == len) && (osNums.str().size() > 0)))
     {
        if(bRepeated)
        {
           osDump << "*" << std::endl;
           bRepeated = false;
        }
        osDump << std::setbase(16) << std::setw(8) << std::setfill('0') << (i - (i % w)) << "  "
              << std::setfill(' ') << std::setiosflags(std::ios_base::left)
              << std::setw(3 * w + ((w / 8) - 1) * 2) << osNums.str()
              << " |" << osChars.str() << std::resetiosflags(std::ios_base::left) << "|" << std::endl;
        szPrevNums = osNums.str();
        osNums.str("");
        osChars.str("");
     }
  }

  osDump << std::setbase(16) << std::setw(8) << std::setfill('0') << (i-1) << std::endl;

  return osDump.str();
}

