#
# Configuration
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)
PROJECT(TraceTransform)

# Local modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

# Build options
OPTION(USE_BACKWARD "Use libbackward to provide pretty stack traces" OFF)
OPTION(USE_ASAN "Use Clang's address sanitized to detect memory issues" OFF)
IF (USE_ASAN AND USE_BACKWARD)
    MESSAGE(FATAL_ERROR "Address Sanitizer and libbackward are incompatible")
ENDIF()


#
# Compile flags
#

# Standard libraries
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# Extra warnings
IF (${CMAKE_C_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
ENDIF()
IF (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
ENDIF()

# Address Sanitizer
IF (USE_ASAN)
    INCLUDE(CheckCCompilerFlag)
    SET(CMAKE_REQUIRED_FLAGS "-fsanitize=address") 
    CHECK_C_COMPILER_FLAG("" HAVE_ASAN)
    IF (${HAVE_ASAN})
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O1 -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls")
    ELSE()
         MESSAGE(FATAL_ERROR "Address Sanitizer not supported by your compiler")
    ENDIF()
ENDIF()

# Profile build
SET(CMAKE_CXX_FLAGS_PROFILE "-O2 -g -DNDEBUG -fno-omit-frame-pointer -fno-inline")
SET(CMAKE_C_FLAGS_PROFILE "-O2 -g -DNDEBUG -fno-omit-frame-pointer -fno-inline")


#
# Dependencies
#

# Boost
IF (WIN32)
	SET(Boost_USE_STATIC_LIBS ON)
ENDIF (WIN32)
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options system filesystem)
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIR})

# Eigen
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Elfutils
IF (USE_BACKWARD)
    FIND_PACKAGE(Elfutils REQUIRED)
ENDIF()

# Intel MKL
SET(MKL_LINKTOOL_OPTS --linking=dynamic)
FIND_PACKAGE(MKL)
IF (MKL_FOUND)
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MKL_LDFLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MKL_CFLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MKL_CFLAGS}")

    ADD_DEFINITIONS(-DEIGEN_USE_MKL_ALL)
    IF (NOT "${MKL_ENV}" STREQUAL "")
        MESSAGE(STATUS "You will have to define the following environment variables at runtime:\n${MKL_ENV}")
    ENDIF()
ENDIF (MKL_FOUND)

# FFTW
FIND_PACKAGE(FFTW REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${FFTW_INCLUDE_DIR})


#
# Libraries
#

IF (USE_BACKWARD)
	ADD_LIBRARY(backward lib/backward/backward.hpp lib/backward/backward.cpp)
    ADD_DEFINITIONS(-DBACKWARD_HAS_DW=1)
	TARGET_LINK_LIBRARIES(backward ${ELFUTILS_LIBRARIES})
	SET(BACKWARD -Wl,--whole-archive backward -Wl,--no-whole-archive)
ENDIF (USE_BACKWARD)

ADD_LIBRARY(auxiliary src/auxiliary.hpp src/auxiliary.cpp)
ADD_LIBRARY(logger src/logger.hpp src/logger.cpp)
SET(COMMON_LIBRARIES auxiliary logger)

ADD_LIBRARY(functionals src/functionals.h src/functionals.c)
TARGET_LINK_LIBRARIES(functionals ${FFTW_LIBRARIES})

ADD_LIBRARY(sinogram src/sinogram.hpp src/sinogram.cpp)
TARGET_LINK_LIBRARIES(sinogram functionals ${COMMON_LIBRARIES})

ADD_LIBRARY(circus src/circus.hpp src/circus.cpp)
TARGET_LINK_LIBRARIES(circus functionals ${COMMON_LIBRARIES})

ADD_LIBRARY(transform src/transform.hpp src/transform.cpp)
TARGET_LINK_LIBRARIES(transform ${COMMON_LIBRARIES} sinogram circus)


#
# Executables
#

ADD_EXECUTABLE(demo src/demo.cpp)
TARGET_LINK_LIBRARIES(demo ${COMMON_LIBRARIES} transform ${Boost_LIBRARIES})
IF (USE_BACKWARD)
	TARGET_LINK_LIBRARIES(demo debug ${BACKWARD})
ENDIF (USE_BACKWARD)

ADD_EXECUTABLE(rottest src/rottest.cpp)
TARGET_LINK_LIBRARIES(rottest ${COMMON_LIBRARIES})
