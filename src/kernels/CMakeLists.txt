ADD_LIBRARY(common common.hpp common.cpp)

CUDA_ADD_LIBRARY(hello hello.hpp hello.cu)
TARGET_LINK_LIBRARIES(hello common)

SET(KERNELS hello PARENT_SCOPE)