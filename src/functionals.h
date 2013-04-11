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
size_t findWeighedMedian(const double* data, const size_t length);
size_t findWeighedMedianSquared(const double* data, const size_t length);

// T functionals
double TFunctionalRadon(const double* data, const size_t length);
double TFunctional1(const double* data, const size_t length);
double TFunctional2(const double* data, const size_t length);
double TFunctional3(const double* data, const size_t length);
double TFunctional4(const double* data, const size_t length);
double TFunctional5(const double* data, const size_t length);

// P-functionals
double PFunctional1(const double* data, const size_t length);
double PFunctional2(const double* data, const size_t length);
double PFunctional3(const double* data, const size_t length);
double PFunctionalHermite(const double* data, const size_t length, unsigned int order, size_t center);

#endif
