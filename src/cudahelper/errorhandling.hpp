//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_CUDAHELPER_ERRORHANDLING_
#define _TRACETRANSFORM_CUDAHELPER_ERRORHANDLING_

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>

// Local
#include "error.hpp"


//
// Module definitions
//

namespace CUDAHelper
{
     // Check Cuda API return code, throws CudaError
     // if given state indicates an error
     inline void checkError(cudaError_t state)
     {
         if (state != cudaSuccess) {
             throw Error(state);
         }
     }

    // Checks cudaGetLastError() and throws CudaError if error detected
    inline void checkState()
    {
        checkError(cudaGetLastError());
    }
}

#endif
