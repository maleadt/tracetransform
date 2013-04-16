ADD_LIBRARY(common common.h common.c)

CUDA_ADD_LIBRARY(hello hello.h hello.cu)
TARGET_LINK_LIBRARIES(hello common)

SET(KERNELS hello PARENT_SCOPE)