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

# Profile build
SET(CMAKE_CXX_FLAGS_PROFILE "-O2 -g -DNDEBUG -fno-omit-frame-pointer -fno-inline")
SET(CMAKE_C_FLAGS_PROFILE "-O2 -g -DNDEBUG -fno-omit-frame-pointer -fno-inline")


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
IF (NOT ELFUTILS_FOUND)
	MESSAGE(STATUS "Not using fancy stack traces (install elfutils to enable).")
ENDIF (NOT ELFUTILS_FOUND)

# CUDA
# FIXME: Don't propagate host flags, since nvcc doesn't support C++11.
#        This makes that we manually need to pass some stuff (such as -g).
SET(CUDA_PROPAGATE_HOST_FLAGS FALSE)
IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -g -G")
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Debug")
IF (CMAKE_BUILD_TYPE STREQUAL "Profile")
	SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -g --generate-line-info")
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Profile")
FIND_PACKAGE(CUDA)
INCLUDE_DIRECTORIES(SYSTEM ${CUDA_INCLUDE_DIRS})

# CULA
FIND_PACKAGE(CULA REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${CULA_INCLUDE_DIR})


#
# Libraries
#

ADD_SUBDIRECTORY(src/kernels)

IF (ELFUTILS_FOUND)
	ADD_LIBRARY(backward lib/backward/backward.hpp lib/backward/backward.cpp)
    ADD_DEFINITIONS(-DBACKWARD_HAS_DW=1)
	TARGET_LINK_LIBRARIES(backward ${ELFUTILS_LIBRARIES})
	SET(BACKWARD -Wl,--whole-archive backward -Wl,--no-whole-archive)
ENDIF (ELFUTILS_FOUND)

ADD_LIBRARY(auxiliary src/auxiliary.hpp src/auxiliary.cpp)
ADD_LIBRARY(logger src/logger.hpp src/logger.cpp)
SET(COMMON_LIBRARIES auxiliary logger)

ADD_LIBRARY(sinogram src/sinogram.hpp src/sinogram.cpp)
TARGET_LINK_LIBRARIES(sinogram ${KERNELS} ${COMMON_LIBRARIES})

ADD_LIBRARY(circus src/circus.hpp src/circus.cpp)
TARGET_LINK_LIBRARIES(circus ${KERNELS} ${COMMON_LIBRARIES})

ADD_LIBRARY(transform src/transform.hpp src/transform.cpp)
TARGET_LINK_LIBRARIES(transform ${COMMON_LIBRARIES} sinogram circus)


#
# Executables
#

ADD_EXECUTABLE(demo src/demo.cpp)
TARGET_LINK_LIBRARIES(demo ${COMMON_LIBRARIES} transform ${Boost_LIBRARIES})
IF (ELFUTILS_FOUND)
	TARGET_LINK_LIBRARIES(demo debug ${BACKWARD})
ENDIF (ELFUTILS_FOUND)

ADD_EXECUTABLE(rottest src/rottest.cpp)
TARGET_LINK_LIBRARIES(rottest ${KERNELS} ${COMMON_LIBRARIES})
