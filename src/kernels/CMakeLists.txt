ADD_LIBRARY(common common.hpp common.cpp)

CUDA_ADD_LIBRARY(hello hello.hpp hello.cu)
TARGET_LINK_LIBRARIES(hello common)

CUDA_ADD_LIBRARY(rotate rotate.hpp rotate.cu)
TARGET_LINK_LIBRARIES(rotate common)

SET(KERNELS hello rotate PARENT_SCOPE)