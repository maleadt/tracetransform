//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_FUNCTIONALS_
#define _TRACETRANSFORM_FUNCTIONALS_

// Standard library
#include <string.h>


//
// Auxiliary
//

size_t findWeighedMedian(const float* data, const size_t length);
size_t findWeighedMedianSquared(const float* data, const size_t length);

//
// T functionals
//

// Radon
float TFunctionalRadon(const float* data, const size_t length);

// T1
float TFunctional1(const float* data, const size_t length);

// T2
float TFunctional2(const float* data, const size_t length);

// T3, T4 and T5
typedef struct {
        float *real;
        float *imag;
} TFunctional345_precalc_t;
TFunctional345_precalc_t *TFunctional3_prepare(size_t rows, size_t cols);
TFunctional345_precalc_t *TFunctional4_prepare(size_t rows, size_t cols);
TFunctional345_precalc_t *TFunctional5_prepare(size_t rows, size_t cols);
float TFunctional345(const float* data, const size_t length, TFunctional345_precalc_t *precalc);
void TFunctional345_destroy(TFunctional345_precalc_t *precalc);


//
// P-functionals
//

// P1
float PFunctional1(const float* data, const size_t length);

// P2
float PFunctional2(const float* data, const size_t length);

// P3
float PFunctional3(const float* data, const size_t length);

// Hermite P-functionals
float PFunctionalHermite(const float* data, const size_t length, unsigned int order, size_t center);

#endif
