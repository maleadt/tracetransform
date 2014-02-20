CUDA_ADD_LIBRARY(rotate rotate.hpp rotate.cu)

CUDA_ADD_LIBRARY(functionals functionals.hpp functionals.cu)
TARGET_LINK_LIBRARIES(functionals cufft)

CUDA_ADD_LIBRARY(stats stats.hpp stats.cu)

SET(KERNELS rotate functionals stats PARENT_SCOPE)

IF (${CULA_FOUND})
    CUDA_ADD_LIBRARY(nos nos.hpp nos.cu)
    TARGET_LINK_LIBRARIES(nos ${CULA_LIBRARIES})

    SET(KERNELS ${KERNELS} nos PARENT_SCOPE)
ENDIF()
