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
extern void TFunctional345(const CUDAHelper::GlobalMemory<float> *input,
                const CUDAHelper::GlobalMemory<float> *precalc_real,
                const CUDAHelper::GlobalMemory<float> *precalc_imag,
                CUDAHelper::GlobalMemory<float> *output, int a);

// P-functionals
extern void PFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output);
extern void PFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output);
extern void PFunctional3(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output);
extern void PFunctionalHermite(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, unsigned int order,
                int center);

#endif
