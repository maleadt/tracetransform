#
# Configuration
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)
PROJECT(TraceTransform)

# Local modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

# Standard libraries
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# Extra warnings
IF (${CMAKE_C_COMPILER_ID})
    IF (${CMAKE_C_COMPILER_ID} MATCHES "GCC" OR ${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
    ENDIF()
    IF (${CMAKE_CXX_COMPILER_ID} MATCHES "GCC" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    ENDIF()
ENDIF()

# Add address sanitizer to debug build
IF (${CMAKE_C_COMPILER_ID})
    IF (${CMAKE_C_COMPILER_ID} MATCHES "Clang" AND NOT ${CMAKE_C_COMPILER_VERSION} LESS "3.2")
        SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1 -fsanitize=address -fno-omit-frame-pointer") 
    ENDIF()
    IF (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang" AND NOT ${CMAKE_CXX_COMPILER_VERSION} LESS "3.2")
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1 -fsanitize=address -fno-omit-frame-pointer") 
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
FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options)
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIR})

# Eigen
FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Elfutils
FIND_PACKAGE(Elfutils)
IF (Elfutils_FOUND)
	ADD_DEFINITIONS(-DBACKWARD_HAS_DW=1)
ENDIF (Elfutils_FOUND)

# Intel MKL
# TODO: -l static
IF (DEFINED ENV{MKLROOT})
    FIND_PROGRAM(LINKTOOL mkl_link_tool
        PATHS $ENV{MKLROOT}/tools)
    if(NOT LINKTOOL)
      MESSAGE(FATAL_ERROR "Could not find 'mkl_link_tool'; verify your MKL installation")
    ENDIF(NOT LINKTOOL)

    EXECUTE_PROCESS(COMMAND ${LINKTOOL} -check_mkl_presence -libs
        OUTPUT_VARIABLE MKL_LDFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE MKL_ERROR
        RESULT_VARIABLE RETVAR)
    IF (NOT "${RETVAR}" STREQUAL "0")
      message(FATAL_ERROR "MKL link tool failure while detecting library flags\n${MKL_ERROR}")
    ENDIF()

    EXECUTE_PROCESS(COMMAND ${LINKTOOL} -check_mkl_presence -opts
        OUTPUT_VARIABLE MKL_CFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE MKL_ERROR
        RESULT_VARIABLE RETVAR)
    IF (NOT "${RETVAR}" STREQUAL "0")
      message(FATAL_ERROR "MKL link tool failure while detecting compilation flags\n${MKL_ERROR}")
    ENDIF()

    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MKL_LDFLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MKL_CFLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MKL_CFLAGS}")

    EXECUTE_PROCESS(COMMAND ${LINKTOOL} -check_mkl_presence -env
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE MKL_ENV
        ERROR_VARIABLE MKL_ERROR
        RESULT_VARIABLE RETVAR)
    IF (NOT "${RETVAR}" STREQUAL "0")
      message(FATAL_ERROR "MKL link tool failure while detecting environment flags\n${MKL_ERROR}")
    ENDIF()

    IF (NOT "${MKL_ENV}" STREQUAL "")
        MESSAGE(STATUS "You will have to define the following environment variables at runtime:\n${MKL_ENV}")
    ENDIF()

    ADD_DEFINITIONS(-DEIGEN_USE_MKL_ALL)
ELSE()
    MESSAGE(STATUS "Not using Intel MKL (specify MKLROOT to do so).")
ENDIF()

# OpenMP
FIND_PACKAGE(OpenMP REQUIRED)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")


#
# Libraries
#

IF (Elfutils_FOUND)
	ADD_LIBRARY(backward lib/backward/backward.hpp lib/backward/backward.cpp)
	TARGET_LINK_LIBRARIES(backward ${DW_LIBRARY})
	SET(BACKWARD -Wl,--whole-archive backward -Wl,--no-whole-archive)
ENDIF (Elfutils_FOUND)

ADD_LIBRARY(auxiliary src/auxiliary.hpp src/auxiliary.cpp)
ADD_LIBRARY(logger src/logger.hpp src/logger.cpp)
SET(COMMON_LIBRARIES auxiliary logger)

ADD_LIBRARY(functionals src/functionals.h src/functionals.c)

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
IF (Elfutils_FOUND)
	TARGET_LINK_LIBRARIES(demo debug ${BACKWARD})
ENDIF (Elfutils_FOUND)

ADD_EXECUTABLE(rottest src/rottest.cpp)
TARGET_LINK_LIBRARIES(rottest ${COMMON_LIBRARIES})
