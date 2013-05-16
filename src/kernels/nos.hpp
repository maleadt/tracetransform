//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_KERNEL_NOS_
#define _TRACETRANSFORM_KERNEL_NOS_

// Local
#include "../cudahelper/memory.hpp"


//
// Routines
//

extern CUDAHelper::GlobalMemory<float> *nearest_orthonormal_sinogram(
                const CUDAHelper::GlobalMemory<float> *input,
                size_t &new_center);

#endif
