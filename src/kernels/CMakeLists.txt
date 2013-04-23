ADD_LIBRARY(common common.hpp common.cpp)

CUDA_ADD_LIBRARY(rotate rotate.hpp rotate.cu)
TARGET_LINK_LIBRARIES(rotate common)

CUDA_ADD_LIBRARY(gpufunctionals functionals.hpp functionals.cu)
TARGET_LINK_LIBRARIES(gpufunctionals common)

SET(KERNELS rotate gpufunctionals PARENT_SCOPE)