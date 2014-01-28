ADD_LIBRARY(common common.hpp common.cpp)

CUDA_ADD_LIBRARY(rotate rotate.hpp rotate.cu)
TARGET_LINK_LIBRARIES(rotate common)

CUDA_ADD_LIBRARY(functionals functionals.hpp functionals.cu)
TARGET_LINK_LIBRARIES(functionals common ${ArrayFire_LIBRARIES})

SET(KERNELS rotate functionals PARENT_SCOPE)

IF (${CULA_FOUND})
    CUDA_ADD_LIBRARY(nos nos.hpp nos.cu)
    TARGET_LINK_LIBRARIES(nos common ${CULA_LIBRARIES})

    SET(KERNELS ${KERNELS} nos PARENT_SCOPE)
ENDIF()
