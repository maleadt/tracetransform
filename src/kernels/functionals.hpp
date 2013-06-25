//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_KERNELS_FUNCTIONALS_
#define _TRACETRANSFORM_KERNELS_FUNCTIONALS_

// Local
#include "../cudahelper/memory.hpp"

//
// T functionals
//

// Radon
extern void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output,
                int a);

// T1
typedef struct {
        CUDAHelper::GlobalMemory<float> *prescan;
        CUDAHelper::GlobalMemory<int> *medians;
} TFunctional12_precalc_t;
TFunctional12_precalc_t *TFunctional12_prepare(size_t rows, size_t cols);
extern void TFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                TFunctional12_precalc_t *precalc,
                CUDAHelper::GlobalMemory<float> *output, int a);
void TFunctional12_destroy(TFunctional12_precalc_t *precalc);

// T2
extern void TFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                TFunctional12_precalc_t *precalc,
                CUDAHelper::GlobalMemory<float> *output, int a);

/// T3, T4 and T5
typedef struct {
        CUDAHelper::GlobalMemory<float> *real;
        CUDAHelper::GlobalMemory<float> *imag;

        CUDAHelper::GlobalMemory<float> *prescan;
        CUDAHelper::GlobalMemory<int> *medians;
} TFunctional345_precalc_t;
TFunctional345_precalc_t *TFunctional3_prepare(size_t rows, size_t cols);
TFunctional345_precalc_t *TFunctional4_prepare(size_t rows, size_t cols);
TFunctional345_precalc_t *TFunctional5_prepare(size_t rows, size_t cols);
extern void TFunctional345(const CUDAHelper::GlobalMemory<float> *input,
                TFunctional345_precalc_t *precalc,
                CUDAHelper::GlobalMemory<float> *output, int a);
void TFunctional345_destroy(TFunctional345_precalc_t *precalc);

//
// P-functionals
//

// P1
extern void PFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output);

// P2
extern void PFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output);

// P3
extern void PFunctional3(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output);

// Hermite P-functionals
extern void PFunctionalHermite(const CUDAHelper::GlobalMemory<float> *input,
                CUDAHelper::GlobalMemory<float> *output, unsigned int order,
                int center);

#endif
