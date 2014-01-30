//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_KERNEL_STATS_
#define _TRACETRANSFORM_KERNEL_STATS_

// Local
#include "../cudahelper/memory.hpp"


//
// Routines
//

CUDAHelper::GlobalMemory<float> *
zscore(const CUDAHelper::GlobalMemory<float> *data);

#endif
