#
# Configuration
#

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.10)
PROJECT(TraceTransform)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-op")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnormalized=nfc")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wtrampolines")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wundef")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wvla")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunreachable-code")

# Test execution string
SEPARATE_ARGUMENTS(TEST_EXECUTION UNIX_COMMAND "-T 1 -P H1 ../res/Cam1_V1.pgm")

# Add address sanitizer to debug build
IF (${CMAKE_C_COMPILER_ID} MATCHES "Clang" AND NOT ${CMAKE_C_COMPILER_VERSION} LESS "3.2")
    SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1 -fsanitize=address -fno-omit-frame-pointer") 
ENDIF()
IF (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang" AND NOT ${CMAKE_CXX_COMPILER_VERSION} LESS "3.2")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1 -fsanitize=address -fno-omit-frame-pointer") 
ENDIF()

# Profile build
SET(CMAKE_CXX_FLAGS_PROFILE "-O1 -g")
SET(CMAKE_C_FLAGS_PROFILE "-O1 -g")

# Profile target
ADD_CUSTOM_TARGET(profile
    COMMAND valgrind --tool=callgrind ./transform ${TEST_EXECUTION}
    DEPENDS transform)

#
# Dependencies
#

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++0x")

FIND_PACKAGE(Boost REQUIRED COMPONENTS program_options)
INCLUDE_DIRECTORIES(SYSTEM ${Boost_INCLUDE_DIR})

FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${EIGEN3_INCLUDE_DIR})


#
# Libraries
#

ADD_LIBRARY(functionals src/functionals.h src/functionals.c)
ADD_EXECUTABLE(transform src/main.cpp)
TARGET_LINK_LIBRARIES(transform functionals ${OpenCV_LIBS} ${Boost_LIBRARIES})


#
# Executables
#

ADD_EXECUTABLE(test_tutorial src/test_tutorial.cpp)
TARGET_LINK_LIBRARIES(test_tutorial functionals ${OpenCV_LIBS})
