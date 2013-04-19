//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_KERNEL_ROTATE_
#define _TRACETRANSFORM_KERNEL_ROTATE_

// Local
#include "../cudahelper/memory.hpp"

// Kernel declarations
extern CUDAHelper::GlobalMemory<float> *rotate(
                const CUDAHelper::GlobalMemory<float> *input, float angle,
                int rows, int cols);
#endif
