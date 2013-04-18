//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_FUNCTIONALS_
#define _TRACETRANSFORM_FUNCTIONALS_

// Standard library
#include <string.h>


//
// Routines
//

// Auxiliary
size_t findWeighedMedian(const float* data, const size_t length);
size_t findWeighedMedianSquared(const float* data, const size_t length);

// T functionals
float TFunctionalRadon(const float* data, const size_t length);
float TFunctional1(const float* data, const size_t length);
float TFunctional2(const float* data, const size_t length);
float TFunctional3(const float* data, const size_t length);
float TFunctional4(const float* data, const size_t length);
float TFunctional5(const float* data, const size_t length);

// P-functionals
float PFunctional1(const float* data, const size_t length);
float PFunctional2(const float* data, const size_t length);
float PFunctional3(const float* data, const size_t length);
float PFunctionalHermite(const float* data, const size_t length, unsigned int order, size_t center);

#endif
