//
// Configuration
//

// Header include
#include "functionals.h"

// Standard library
#include <complex.h>
#include <math.h>
#include <stdlib.h>

// M_PI is dropped in GCC's C99
#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif



////////////////////////////////////////////////////////////////////////////////
// Auxiliary
//

size_t findWeighedMedian(const float* data, const size_t length)
{               
        float sum = 0;
        for (size_t i = 0; i < length; i++)
                sum += data[i];
        float integral = 0;
        for (size_t i = 0; i < length; i++) {
                integral += data[i];
                if (2*integral >= sum)
                        return i;
        }
        return length-1;
}

size_t findWeighedMedianSqrt(const float* data, const size_t length)
{               
        float sum = 0;
        for (size_t i = 0; i < length; i++)
                sum += sqrt(data[i]);

        float integral = 0;
        for (size_t i = 0; i < length; i++) {
                integral += sqrt(data[i]);
                if (2*integral >= sum)
                        return i;
        }

        return length-1;
}

float trapz(const float* x, const float* y, const size_t length)
{
        float sum = 0;
        for (size_t i = 0; i < length-1; i++) {
                sum += (x[i+1] - x[i]) * (y[i+1] + y[i]);
        }
        return sum * 0.5;
}

float hermite_polynomial(unsigned int order, float x) {
        switch (order) {
                case 0:
                        return 1.0;

                case 1:
                        return 2.0*x;

                default:
                        return 2.0*x*hermite_polynomial(order-1, x)
                                -2.0*(order-1)*hermite_polynomial(order-2, x);
        }
}

unsigned int factorial(unsigned int n)
{
        return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

float hermite_function(unsigned int order, float x) {
        return hermite_polynomial(order, x) / (
                        sqrt(pow(2, order) * factorial(order) * sqrt(M_PI))
                        * exp(x*x/2)
                );
}



////////////////////////////////////////////////////////////////////////////////
// T-functionals
//

//
// Radon
//

float TFunctionalRadon(const float* data, const size_t length)
{
        float integral = 0;
        for (size_t t = 0; t < length; t++)
                integral += data[t];
        return integral;                
}


//
// T1
//

float TFunctional1(const float* data, const size_t length)
{
        // Transform the domain from t to r
        size_t median = findWeighedMedian(data, length);

        // Integrate
        float integral = 0;
        for (size_t r = 0; r < length-median; r++)
                integral += data[r+median] * r;
        return integral;                
}


//
// T2
//

float TFunctional2(const float* data, const size_t length)
{
        // Transform the domain from t to r
        size_t median = findWeighedMedian(data, length);

        // Integrate
        float integral = 0;
        for (size_t r = 0; r < length-median; r++)
                integral += data[r+median] * r*r;
        return integral;                
}


//
// T3, T4 and T5
//

TFunctional345_precalc_t *TFunctional3_prepare(size_t length)
{
        TFunctional345_precalc_t *precalc = (TFunctional345_precalc_t*) malloc(sizeof(TFunctional345_precalc_t));

        precalc->real = (float*) malloc(length*sizeof(float));
        precalc->imag = (float*) malloc(length*sizeof(float));

        for (unsigned int r = 1; r < length; r++) {
                precalc->real[r] = r*cos(5.0*log(r));
                precalc->imag[r] = r*sin(5.0*log(r));
        }

        return precalc;
}

TFunctional345_precalc_t *TFunctional4_prepare(size_t length)
{
        TFunctional345_precalc_t *precalc = (TFunctional345_precalc_t*) malloc(sizeof(TFunctional345_precalc_t));

        precalc->real = (float*) malloc(length*sizeof(float));
        precalc->imag = (float*) malloc(length*sizeof(float));

        for (unsigned int r = 1; r < length; r++) {
                precalc->real[r] = cos(3.0*log(r));
                precalc->imag[r] = sin(3.0*log(r));
        }

        return precalc;
}

TFunctional345_precalc_t *TFunctional5_prepare(size_t length)
{
        TFunctional345_precalc_t *precalc = (TFunctional345_precalc_t*) malloc(sizeof(TFunctional345_precalc_t));

        precalc->real = (float*) malloc(length*sizeof(float));
        precalc->imag = (float*) malloc(length*sizeof(float));

        for (unsigned int r = 1; r < length; r++) {
                precalc->real[r] = sqrt(r)*cos(4.0*log(r));
                precalc->imag[r] = sqrt(r)*sin(4.0*log(r));
        }

        return precalc;
}

float TFunctional345(const float* data, const size_t length, TFunctional345_precalc_t *precalc)
{
        // Transform the domain from t to r1
        size_t squaredmedian = findWeighedMedianSqrt(data, length);

        // Integrate
        float integral_real = 0, integral_imag = 0;
        for (size_t r1 = 1; r1 < length-squaredmedian; r1++) {
                // From 1, since exp(i*log(0)) == 0
                integral_real += precalc->real[r1] * data[r1+squaredmedian];
                integral_imag += precalc->imag[r1] * data[r1+squaredmedian];

        }
        return hypot(integral_real, integral_imag);
}

void TFunctional345_destroy(TFunctional345_precalc_t *precalc)
{
        free(precalc->real);
        free(precalc->imag);
        free(precalc);
}



////////////////////////////////////////////////////////////////////////////////
// P-functionals
//

//
// P1
//

float PFunctional1(const float* data, const size_t length)
{
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

float PFunctional2(const float* data, const size_t length)
{
        size_t median = findWeighedMedian(data, length);
        return data[median];
}

//
// P3
//

float PFunctional3(const float* data, const size_t length)
{
        // Calculate the discrete Fourier transform
        float *fourier_real = (float*) calloc(length, sizeof(float));
        float *fourier_imag = (float*) calloc(length, sizeof(float));
        for (size_t i = 0; i < length; i++) {
                fourier_real[i] = 0;
                fourier_imag[i] = 0;
                float arg = -2.0 * M_PI * (float)i / (float)length;
                for (size_t j = 0; j < length; j++) {
                        float cosarg = cos(j * arg);
                        float sinarg = sin(j * arg);
                        fourier_real[i] += data[j] * cosarg;
                        fourier_imag[i] += data[j] * sinarg;
                }
        }

        // Integrate
        // NOTE: we abuse previously allocated vectors fourier_real and
        //       fourier_imag to respectively save the linear space (x) and
        //       modifier Fourier values (y)
        for (size_t p = 0; p < length; p++) {
                fourier_imag[p] = pow(hypot(fourier_real[p]/length, fourier_imag[p]/length), 4);
                fourier_real[p] = -1 + p*2.0/(length-1);
        }
        float sum = trapz(fourier_real, fourier_imag, length);
        free(fourier_real);
        free(fourier_imag);
        return sum;
}

//
// Hermite P-functionals
//

float PFunctionalHermite(const float* data, const size_t length, unsigned int order, size_t center)
{
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
                        z = (p-center) * stepsize_upper;
                integral += data[p] * hermite_function(order, z);
        }
        return integral;
}
