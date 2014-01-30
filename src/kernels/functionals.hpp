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
void TFunctionalRadon(const CUDAHelper::GlobalMemory<float> *input,
                      CUDAHelper::GlobalMemory<float> *output, int a);

// T1
typedef struct {
    CUDAHelper::GlobalMemory<float> *prescan;
    CUDAHelper::GlobalMemory<int> *medians;
} TFunctional12_precalc_t;
TFunctional12_precalc_t *TFunctional12_prepare(size_t rows, size_t cols);
void TFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                  TFunctional12_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output, int a);
void TFunctional12_destroy(TFunctional12_precalc_t *precalc);

// T2
void TFunctional2(const CUDAHelper::GlobalMemory<float> *input,
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
void TFunctional345(const CUDAHelper::GlobalMemory<float> *input,
                    TFunctional345_precalc_t *precalc,
                    CUDAHelper::GlobalMemory<float> *output, int a);
void TFunctional345_destroy(TFunctional345_precalc_t *precalc);

// T6
typedef struct {
    CUDAHelper::GlobalMemory<float> *prescan;
    CUDAHelper::GlobalMemory<int> *medians;
    CUDAHelper::GlobalMemory<float> *extracted;
    CUDAHelper::GlobalMemory<float> *weighted;
    CUDAHelper::GlobalMemory<int> *indices;
    CUDAHelper::GlobalMemory<float> *permuted;
} TFunctional6_precalc_t;
TFunctional6_precalc_t *TFunctional6_prepare(size_t rows, size_t cols);
void TFunctional6(const CUDAHelper::GlobalMemory<float> *input,
                  TFunctional6_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output, int a);
void TFunctional6_destroy(TFunctional6_precalc_t *precalc);

// T7
typedef struct {
    CUDAHelper::GlobalMemory<float> *prescan;
    CUDAHelper::GlobalMemory<int> *medians;
    CUDAHelper::GlobalMemory<float> *extracted;
} TFunctional7_precalc_t;
TFunctional7_precalc_t *TFunctional7_prepare(size_t rows, size_t cols);
void TFunctional7(const CUDAHelper::GlobalMemory<float> *input,
                  TFunctional7_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output, int a);
void TFunctional7_destroy(TFunctional7_precalc_t *precalc);


//
// P-functionals
//

// P1
void PFunctional1(const CUDAHelper::GlobalMemory<float> *input,
                  CUDAHelper::GlobalMemory<float> *output);

// P2
typedef struct {
    CUDAHelper::GlobalMemory<float> *prescan;
    CUDAHelper::GlobalMemory<int> *medians;
} PFunctional2_precalc_t;
PFunctional2_precalc_t *PFunctional2_prepare(size_t rows, size_t cols);
void PFunctional2(const CUDAHelper::GlobalMemory<float> *input,
                  PFunctional2_precalc_t *precalc,
                  CUDAHelper::GlobalMemory<float> *output);
void PFunctional2_destroy(PFunctional2_precalc_t *precalc);

// P3
void PFunctional3(const CUDAHelper::GlobalMemory<float> *input,
                  CUDAHelper::GlobalMemory<float> *output);

#ifdef WITH_CULA
// Hermite P-functionals
void PFunctionalHermite(const CUDAHelper::GlobalMemory<float> *input,
                        CUDAHelper::GlobalMemory<float> *output,
                        unsigned int order, int center);
#endif

#endif
