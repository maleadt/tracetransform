//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_KERNEL_ROTATE_
#define _TRACETRANSFORM_KERNEL_ROTATE_

// Local
#include "../cudahelper/memory.hpp"


//
// Routines
//

extern void rotate(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, float angle);
#endif
