//
// Configuration
//

// Header include
#include "functionals.hpp"

// Standard library
#include <cmath>   // for log, sqrt, cos, sin, hypot, etc
#include <cstdlib> // for calloc, qsort, qsort_r
#include <cassert>

// FFTW
#include <fftw3.h>

// OpenMP
#include <omp.h>


////////////////////////////////////////////////////////////////////////////////
// Auxiliary
//

int findWeightedMedian(const Eigen::VectorXf& data) {
    float sum = data.sum();
    float integral = 0;
    for (int i = 0; i < data.size(); i++) {
        integral += data[i];
        if (2 * integral >= sum)
            return i;
    }
    return data.size() - 1;
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
    int *x = (int *)a;
    int *y = (int *)b;
    float *data_weighted = (float *)arg;

    return compareFloat(data_weighted + *x, data_weighted + *y);
}

float trapz(const Eigen::VectorXf &x, const Eigen::VectorXf y) {
    assert(x.size() == y.size());
    float sum = 0;
    for (int i = 0; i < x.size() - 1; i++) {
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

float TFunctionalRadon(const Eigen::VectorXf& data) {
    return data.sum();
}


//
// T1
//

float TFunctional1(const Eigen::VectorXf& data) {
    // Transform the domain from t to r
    int median = findWeightedMedian(data);

    // Integrate
    float integral = 0;
    for (int r = 0; r < data.size() - median; r++)
        integral += data[r + median] * r;
    return integral;
}


//
// T2
//

float TFunctional2(const Eigen::VectorXf& data) {
    // Transform the domain from t to r
    int median = findWeightedMedian(data);

    // Integrate
    float integral = 0;
    for (int r = 0; r < data.size() - median; r++)
        integral += data[r + median] * r * r;
    return integral;
}


//
// T3, T4 and T5
//

TFunctional345_precalc_t *TFunctional3_prepare(int rows,
                                               int) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    precalc->real = (float *)malloc(rows * sizeof(float));
    precalc->imag = (float *)malloc(rows * sizeof(float));

    for (int r = 1; r < rows; r++) {
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

TFunctional345_precalc_t *TFunctional4_prepare(int rows,
                                               int) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    precalc->real = (float *)malloc(rows * sizeof(float));
    precalc->imag = (float *)malloc(rows * sizeof(float));

    for (int r = 1; r < rows; r++) {
#if defined(__clang__)
        sincosf(3.0 * log(r), &precalc->imag[r], &precalc->real[r]);
#else
        precalc->real[r] = cos(3.0 * log(r));
        precalc->imag[r] = sin(3.0 * log(r));
#endif
    }

    return precalc;
}

TFunctional345_precalc_t *TFunctional5_prepare(int rows,
                                               int) {
    TFunctional345_precalc_t *precalc =
        (TFunctional345_precalc_t *)malloc(sizeof(TFunctional345_precalc_t));

    precalc->real = (float *)malloc(rows * sizeof(float));
    precalc->imag = (float *)malloc(rows * sizeof(float));

    for (int r = 1; r < rows; r++) {
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

float TFunctional345(const Eigen::VectorXf& data,
                     TFunctional345_precalc_t *precalc) {
    // Transform the domain from t to r1
    int squaredmedian = findWeightedMedian(data.cwiseSqrt());

    // Integrate
    float integral_real = 0, integral_imag = 0;
    for (int r1 = 1; r1 < data.size() - squaredmedian; r1++) {
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

float TFunctional6(const Eigen::VectorXf& data) {
    // Transform the domain from t to r1
    int squaredmedian = findWeightedMedian(data.cwiseSqrt());
    int length_r1 = data.size() - squaredmedian;

    // Extract and weight data from the positive domain of r1
    // and prepare the indexing array
    Eigen::VectorXi data_weighted_index(length_r1);
    Eigen::VectorXf data_weighted(length_r1);
    for (int r1 = 0; r1 < length_r1; r1++) {
        data_weighted[r1] = (float)r1 * data[r1 + squaredmedian];
        data_weighted_index[r1] = r1;
    }

    // Sort the weighted data
    // NOTE: since we need the indexes later on, we don't actually sort the
    //       array but save the indexes of the sorted elements
    qsort_r(data_weighted_index.data(), length_r1, sizeof(int),
            compareIndexedFloat, data_weighted.data());

    // Permuting the input data
    Eigen::VectorXf data_sort(length_r1);
    for (int r1 = 0; r1 < length_r1; r1++) {
        data_sort[r1] = data[squaredmedian + data_weighted_index[r1]];
    }

    // Weighted median
    int index = findWeightedMedian(data_sort.cwiseSqrt());
    return data_weighted[data_weighted_index[index]];
}


//
// T7
//

float TFunctional7(const Eigen::VectorXf& data) {
    // Transform the domain from t to r
    int median = findWeightedMedian(data);
    int length_r = data.size() - median;

    // Extract data from the positive domain of r
    Eigen::VectorXf data_r(data.tail(length_r));

    // Sorting the transformed data
    qsort(data_r.data(), length_r, sizeof(float), compareFloat);

    // Weighted median
    int index = findWeightedMedian(data_r.cwiseSqrt());
    return data_r[index];
}


////////////////////////////////////////////////////////////////////////////////
// P-functionals
//

//
// P1
//

float PFunctional1(const Eigen::VectorXf& data) {
    float sum = 0;
    float previous = data[0];
    for (int p = 1; p < data.size(); p++) {
        float current = data[p];
        sum += fabs(previous - current);
        previous = current;
    }
    return sum;
}


//
// P2
//

float PFunctional2(const Eigen::VectorXf& data) {
    // Sorting the data
    Eigen::VectorXf sorted(data);
    qsort(sorted.data(), sorted.size(), sizeof(float), compareFloat);

    // Find the weighted median
    int median = findWeightedMedian(sorted);
    return sorted[median];
}


//
// P3
//

PFunctional3_precalc_t *PFunctional3_prepare(int rows) {
    // Enable FFTW multithreading
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    // Plan the DFT we'll need
    float *data = (float*) malloc(sizeof(float)*rows);
    fftwf_complex *fourier = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * rows);
    fftwf_plan p = fftwf_plan_dft_r2c_1d(rows, data, fourier, FFTW_MEASURE);
    assert(p != NULL);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    fftwf_free(fourier);
    free(data);

    // Clean-up FFTW multithreading remnants
    fftw_cleanup_threads();

    return NULL;
}

float PFunctional3(const Eigen::VectorXf& data) {
    // Calculate the discrete Fourier transform
    // TODO: we should only plan this once, given that we use the same array
    fftwf_complex *fourier = (fftwf_complex*) fftwf_malloc(sizeof(fftw_complex) * data.size());
    // NOTE: we can safely cast the const away, because in FFTW_ESTIMATE regime
    //       the input data will be preserved
    fftwf_plan p;
#pragma omp critical (make_plan)
    p = fftwf_plan_dft_r2c_1d(data.size(), (float*)data.data(), fourier, FFTW_ESTIMATE);
    assert(p != NULL);
    fftwf_execute(p);

    // Integrate
    Eigen::VectorXf linspace(data.size());
    Eigen::VectorXf modifier(data.size());
    for (int p = 0; p < data.size()/2+1; p++) {
        modifier[p] =
            pow(hypot(fourier[p][0] / data.size(), fourier[p][1] / data.size()), 4);
        linspace[p] = -1 + p * 2.0 / (data.size() - 1);
    }
    for (int p = data.size()/2+1; p < data.size(); p++) {
        modifier[p] =
            pow(hypot(fourier[data.size()-p][0] / data.size(), fourier[data.size()-p][1] / data.size()), 4);
        linspace[p] = -1 + p * 2.0 / (data.size() - 1);
    }
    float sum = trapz(linspace, modifier);
    fftwf_destroy_plan(p);
    fftwf_free(fourier);
    return sum;
}

void PFunctional3_destroy(PFunctional3_precalc_t *) { return; }


//
// Hermite P-functionals
//

float PFunctionalHermite(const Eigen::VectorXf& data,
                         unsigned int order, int center) {
    // Discretize the [-10, 10] domain to fit the column iterator
    float stepsize_lower = 10.0 / center;
    float stepsize_upper = 10.0 / (data.size() - 1 - center);

    // Calculate the integral
    float integral = 0;
    float z;
    for (int p = 0; p < data.size(); p++) {
        if (p < center)
            z = p * stepsize_lower - 10;
        else
            z = (p - center) * stepsize_upper;
        integral += data[p] * hermite_function(order, z);
    }
    return integral;
}
