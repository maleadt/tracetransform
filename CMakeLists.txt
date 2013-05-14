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
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
SET(CUDA_NVCC_FLAGS "-Xcompiler -Wall,-Wextra -arch sm_35 --ptxas-options=-v")

# Test execution string
SEPARATE_ARGUMENTS(TEST_EXECUTION UNIX_COMMAND "-T 1 -P H1 ../res/Cam1_V1.pgm")

# Profile build
SET(CMAKE_CXX_FLAGS_PROFILE "-O2 -g -DNDEBUG")
SET(CMAKE_C_FLAGS_PROFILE "-O2 -g -DNDEBUG")

# Profile target
ADD_CUSTOM_TARGET(profile
    COMMAND valgrind --tool=callgrind ./transform ${TEST_EXECUTION}
    DEPENDS transform)


#
# Dependencies
#

# C-compiler
IF (NOT ${CMAKE_C_COMPILER_ID} MATCHES "GNU" OR NOT ${CMAKE_C_COMPILER_VERSION} VERSION_LESS "4.7")
    MESSAGE(FATAL_ERROR "CUDA requires GCC <= 4.6")
ENDIF()

# Boost
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

# CUDA
# FIXME: Don't propagate host flags, since nvcc doesn't support C++11.
#        This makes that we manually need to pass some stuff (such as -g).
SET(CUDA_PROPAGATE_HOST_FLAGS FALSE)
IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -g -G")
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")
FIND_PACKAGE(CUDA)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})


#
# Libraries
#

ADD_SUBDIRECTORY(src/kernels)

ADD_LIBRARY(backward lib/backward/backward.hpp lib/backward/backward.cpp)
TARGET_LINK_LIBRARIES(backward ${DW_LIBRARY})
SET(BACKWARD -Wl,--whole-archive backward -Wl,--no-whole-archive)

ADD_LIBRARY(auxiliary src/auxiliary.hpp src/auxiliary.cpp)
ADD_LIBRARY(logger src/logger.hpp src/logger.cpp)
SET(COMMON_LIBRARIES auxiliary logger)

ADD_LIBRARY(functionals src/functionals.h src/functionals.c)

ADD_LIBRARY(sinogram src/sinogram.hpp src/sinogram.cpp)
TARGET_LINK_LIBRARIES(sinogram ${KERNELS} ${COMMON_LIBRARIES})

ADD_LIBRARY(circus src/circus.hpp src/circus.cpp)
TARGET_LINK_LIBRARIES(circus functionals ${KERNELS} ${COMMON_LIBRARIES})

ADD_LIBRARY(transform src/transform.hpp src/transform.cpp)
TARGET_LINK_LIBRARIES(transform ${COMMON_LIBRARIES} sinogram circus)


#
# Executables
#

ADD_EXECUTABLE(demo src/demo.cpp)
TARGET_LINK_LIBRARIES(demo ${COMMON_LIBRARIES} transform ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(demo debug ${BACKWARD})

ADD_EXECUTABLE(benchmark src/benchmark.cpp)
TARGET_LINK_LIBRARIES(benchmark ${COMMON_LIBRARIES} transform ${Boost_LIBRARIES})
