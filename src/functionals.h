//
// Configuration
//

// Include guard
#ifndef FUNCTIONALS_H
#define FUNCTIONALS_H

// System includes
#include <string.h>

// Typedefs
typedef double(*Functional)(const double*, const size_t, const void*);


//
// Routines
//

// Arguments
struct ArgumentsHermite {
	unsigned int order;
	unsigned int center;
};

// Auxiliary
size_t findWeighedMedian(const double* data, const size_t length);
size_t findWeighedMedianSquared(const double* data, const size_t length);

// T functionals
double TFunctionalRadon(const double* data, const size_t length, const void*);
double TFunctional1(const double* data, const size_t length, const void*);
double TFunctional2(const double* data, const size_t length, const void*);
double TFunctional3(const double* data, const size_t length, const void*);
double TFunctional4(const double* data, const size_t length, const void*);
double TFunctional5(const double* data, const size_t length, const void*);

// P-functionals
double PFunctional1(const double* data, const size_t length, const void*);
double PFunctional2(const double* data, const size_t length, const void*);
double PFunctional3(const double* data, const size_t length, const void*);
double PFunctionalHermite(const double* data, const size_t length, const void* _arguments);

#endif
