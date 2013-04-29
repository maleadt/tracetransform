//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_KERNELS_FUNCTIONALS_
#define _TRACETRANSFORM_KERNELS_FUNCTIONALS_

// Local
#include "../cudahelper/memory.hpp"


//
// Routines
//

// T functionals
extern void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output,
                int a);
extern void TFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a);
extern void TFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a);
extern void TFunctional3(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a);
extern void TFunctional4(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a);
extern void TFunctional5(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, int a);

// P-functionals
extern CUDAHelper::GlobalMemory<float> *PFunctional1(
                const CUDAHelper::GlobalMemory<float> *input,
                int a);
extern CUDAHelper::GlobalMemory<float> *PFunctional2(
                const CUDAHelper::GlobalMemory<float> *input,
                int a);
extern CUDAHelper::GlobalMemory<float> *PFunctional3(
                const CUDAHelper::GlobalMemory<float> *input,
                int a);
extern CUDAHelper::GlobalMemory<float> *PFunctionalHermite(
                const CUDAHelper::GlobalMemory<float> *input,
                int a, unsigned int order, int center);

#endif
