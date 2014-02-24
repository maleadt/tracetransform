//
// Configuration
//

// Header include
#include "functionals.h"

// Standard library
#define _GNU_SOURCE // for sincosf
#include <math.h>   // for log, sqrt, cos, sin, hypot, etc
#include <stdlib.h> // for malloc, free, calloc, qsort, qsort_r
#include <string.h> // for memcpy
#include <assert.h>

// FFTW
#include <fftw3.h>

// M_PI is dropped in GCC's C99
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


////////////////////////////////////////////////////////////////////////////////
// Auxiliary
//

size_t findWeightedMedian(const float *data, const size_t length) {
    float sum = 0;
    for (size_t i = 0; i < length; i++)
        sum += data[i];
    float integral = 0;
    for (size_t i = 0; i < length; i++) {
        integral += data[i];
        if (2 * integral >= sum)
            return i;
    }
    return length - 1;
}

size_t findWeightedMedianSqrt(const float *data, const size_t length) {
    float sum = 0;
    for (size_t i = 0; i < length; i++)
        sum += sqrt(data[i]);

    float integral = 0;
    for (size_t i = 0; i < length; i++) {
        integral += sqrt(data[i]);
        if (2 * integral >= sum)
            return i;
    }

    return length - 1;
}

int compareFloat(const void *a, const void *b) {
    float *x = (float *)a;
    float *y = (float *)b;

    if (*x < *y) {
        return -1;
    } else if (*x > *y) {
        return 1;
    } else {
        return 0;
    }
}

int compareIndexedFloat(const void *a, const void *b, void *arg) {
    size_t *x = (size_t *)a;
    size_t *y = (size_t *)b;
    float *data_weighted = (float *)arg;

    return compareFloat(data_weighted + *x, data_weighted + *y);
}

float trapz(const float *x, const float *y, const size_t length) {
    float sum = 0;
    for (size_t i = 0; i < length - 1; i++) {
        sum += (x[i + 1] - x[i]) * (y[i + 1] + y[i]);
    }
    return sum * 0.5;
}

float hermite_polynomial(unsigned int order, float x) {
    switch (order) {
    case 0:
        return 1.0;

    case 1:
        return 2.0 * x;

    default:
        return 2.0 * x * hermite_polynomial(order - 1, x) -
               2.0 * (order - 1) * hermite_polynomial(order - 2, x);
    }
}

unsigned int factorial(unsigned int n) {
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

float hermite_function(unsigned int order, float x) {
    return hermite_polynomial(order, x) /
           (sqrt(pow(2, order) * factorial(order) * sqrt(M_PI)) *
            exp(x * x / 2));
}


////////////////////////////////////////////////////////////////////////////////
// T-functionals
//

//
// Radon
//

float TFunctionalRadon(const float *data, const size_t length) {
    float integral = 0;
    for (size_t t = 0; t < length; t++)
        integral += data[t];
    return integral;
}


//
// T1
//

float TFunctional1(const float *data, const size_t length) {
    // Transform the domain from t to r
    size_t median = findWeightedMedian(data, length);

    // Integrate
    float integral = 0;
    for (size_t r = 0; r < length - median; r++)
        integral += data[r + median] * r;
    return integral;
}


//
// T2
//

float TFunctional2(const float *data, const size_t length) {
    // Transform the domain from t to r
    size_t median = findWeightedMedian(data, length);

    // Integrate
    float integral = 0;
    for (size_t r = 0; r < length - median; r++)
        integral += data[r + median] * r * r;
    return integral;
}


//
// T3, T4 and T5
//

TFunctional345_precalc_t *TFunctional3_prepare(size_t rows, size_t cols) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    precalc->real = (float *)malloc(rows * sizeof(float));
    precalc->imag = (float *)malloc(rows * sizeof(float));

    for (unsigned int r = 1; r < rows; r++) {
#if defined(__clang__)
        sincosf(5.0 * log(r), &precalc->imag[r], &precalc->real[r]);
        precalc->real[r] *= r;
        precalc->imag[r] *= r;
#else
        precalc->real[r] = r * cos(5.0 * log(r));
        precalc->imag[r] = r * sin(5.0 * log(r));
#endif
    }

    return precalc;
}

TFunctional345_precalc_t *TFunctional4_prepare(size_t rows, size_t cols) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    precalc->real = (float *)malloc(rows * sizeof(float));
    precalc->imag = (float *)malloc(rows * sizeof(float));

    for (unsigned int r = 1; r < rows; r++) {
#if defined(__clang__)
        sincosf(3.0 * log(r), &precalc->imag[r], &precalc->real[r]);
        precalc->real[r] *= r;
        precalc->imag[r] *= r;
#else
        precalc->real[r] = cos(3.0 * log(r));
        precalc->imag[r] = sin(3.0 * log(r));
#endif
    }

    return precalc;
}

TFunctional345_precalc_t *TFunctional5_prepare(size_t rows, size_t cols) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    precalc->real = (float *)malloc(rows * sizeof(float));
    precalc->imag = (float *)malloc(rows * sizeof(float));

    for (unsigned int r = 1; r < rows; r++) {
#if defined(__clang__)
        sincosf(4.0 * log(r), &precalc->imag[r], &precalc->real[r]);
        precalc->real[r] *= sqrt(r);
        precalc->imag[r] *= sqrt(r);
#else
        precalc->real[r] = sqrt(r) * cos(4.0 * log(r));
        precalc->imag[r] = sqrt(r) * sin(4.0 * log(r));
#endif
    }

    return precalc;
}

float TFunctional345(const float *data, const size_t length,
                     TFunctional345_precalc_t *precalc) {
    // Transform the domain from t to r1
    size_t squaredmedian = findWeightedMedianSqrt(data, length);

    // Integrate
    float integral_real = 0, integral_imag = 0;
    for (size_t r1 = 1; r1 < length - squaredmedian; r1++) {
        // From 1, since exp(i*log(0)) == 0
        integral_real += precalc->real[r1] * data[r1 + squaredmedian];
        integral_imag += precalc->imag[r1] * data[r1 + squaredmedian];
    }
    return hypot(integral_real, integral_imag);
}

void TFunctional345_destroy(TFunctional345_precalc_t *precalc) {
    free(precalc->real);
    free(precalc->imag);
    free(precalc);
}


//
// T6
//

float TFunctional6(const float *data, const size_t length) {
    // Transform the domain from t to r1
    size_t squaredmedian = findWeightedMedianSqrt(data, length);
    size_t length_r1 = length - squaredmedian;

    // Extract and weight data from the positive domain of r1 and prepare the
    // indexing array
    size_t data_weighted_index[length_r1];
    float data_weighted[length_r1];
    for (size_t r1 = 0; r1 < length_r1; r1++) {
        data_weighted[r1] = (float)r1 * data[r1 + squaredmedian];
        data_weighted_index[r1] = r1;
    }

    // Sort the weighted data
    // NOTE: since we need the indexes later on, we don't actually sort the
    //       array but save the indexes of the sorted elements
    qsort_r(data_weighted_index, length_r1, sizeof(size_t), compareIndexedFloat,
            &data_weighted);

    // Permuting the input data
    float data_sort[length_r1];
    for (size_t r1 = 0; r1 < length_r1; r1++) {
        data_sort[r1] = data[squaredmedian + data_weighted_index[r1]];
    }

    // Weighted median
    size_t index = findWeightedMedianSqrt(data_sort, length_r1);
    return data_weighted[data_weighted_index[index]];
}


//
// T7
//

float TFunctional7(const float *data, const size_t length) {
    // Transform the domain from t to r
    size_t median = findWeightedMedian(data, length);
    size_t length_r = length - median;

    // Extract data from the positive domain of r
    float data_r[length_r];
    memcpy(data_r, data + median, length_r * sizeof(float));

    // Sorting the transformed data
    qsort(data_r, length_r, sizeof(float), compareFloat);

    // Weighted median
    size_t index = findWeightedMedianSqrt(data_r, length_r);
    return data_r[index];
}


////////////////////////////////////////////////////////////////////////////////
// P-functionals
//

//
// P1
//

float PFunctional1(const float *data, const size_t length) {
    float sum = 0;
    float previous = data[0];
    for (size_t p = 1; p < length; p++) {
        float current = data[p];
        sum += fabs(previous - current);
        previous = current;
    }
    return sum;
}


//
// P2
//

float PFunctional2(const float *data, const size_t length) {
    // Sorting the data
    float sorted[length];
    memcpy(sorted, data, length * sizeof(float));
    qsort(sorted, length, sizeof(float), compareFloat);

    // Find the weighted median
    size_t median = findWeightedMedian(sorted, length);
    return sorted[median];
}


//
// P3
//

PFunctional3_precalc_t *PFunctional3_prepare(size_t rows) {
    // Plan the DFT we'll need
    float *data = (float*) malloc(sizeof(float)*rows);
    fftwf_complex *fourier = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * rows);
    fftwf_plan p = fftwf_plan_dft_r2c_1d(rows, data, fourier, FFTW_MEASURE);
    assert(p != NULL);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    fftwf_free(fourier);
    free(data);
}

float PFunctional3(const float *data, const size_t length) {
    // Calculate the discrete Fourier transform
    fftwf_complex *fourier = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * length);
    fftwf_plan p = fftwf_plan_dft_r2c_1d(length, data, fourier, FFTW_ESTIMATE);
    assert(p != NULL);
    fftwf_execute(p);

    // Integrate
    float *linspace = (float*) malloc(sizeof(float) * length);
    float *modifier = (float*) malloc(sizeof(float) * length);
    for (size_t p = 0; p < length/2+1; p++) {
        modifier[p] =
            pow(hypot(fourier[p][0] / length, fourier[p][1] / length), 4);
        linspace[p] = -1 + p * 2.0 / (length - 1);
    }
    for (size_t p = length/2+1; p < length; p++) {
        modifier[p] =
            pow(hypot(fourier[length-p][0] / length, fourier[length-p][1] / length), 4);
        linspace[p] = -1 + p * 2.0 / (length - 1);
    }
    float sum = trapz(linspace, modifier, length);
    fftwf_destroy_plan(p);
    fftwf_free(fourier);
    free(linspace);
    free(modifier);
    return sum;
}

void PFunctional3_destroy(PFunctional3_precalc_t *precalc) {
    return;
}


//
// Hermite P-functionals
//

float PFunctionalHermite(const float *data, const size_t length,
                         unsigned int order, size_t center) {
    // Discretize the [-10, 10] domain to fit the column iterator
    float stepsize_lower = 10.0 / center;
    float stepsize_upper = 10.0 / (length - 1 - center);

    // Calculate the integral
    float integral = 0;
    float z;
    for (size_t p = 0; p < length; p++) {
        if (p < center)
            z = p * stepsize_lower - 10;
        else
            z = (p - center) * stepsize_upper;
        integral += data[p] * hermite_function(order, z);
    }
    return integral;
}
