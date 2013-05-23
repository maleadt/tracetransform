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


//
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


//
// T functionals
//

// T-functional for the Radon transform.
float TFunctionalRadon(const float* data, const size_t length)
{
        float integral = 0;
        for (size_t t = 0; t < length; t++)
                integral += data[t];
        return integral;                
}

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

float TFunctional3(const float* data, const size_t length)
{
        // Transform the domain from t to r1
        size_t squaredmedian = findWeighedMedianSqrt(data, length);

        // Integrate
        float complex integral = 0 + 0*I;
        const float complex factor = 0 + 5*I;
        for (size_t r1 = 1; r1 < length-squaredmedian; r1++) {
                // From 1, since exp(i*log(0)) == 0
                integral += cexp(factor*log(r1))
                        * (r1*data[r1+squaredmedian]);

        }
        return cabs(integral);
}

float TFunctional4(const float* data, const size_t length)
{
        // Transform the domain from t to r1
        size_t squaredmedian = findWeighedMedianSqrt(data, length);

        // Integrate
        float complex integral = 0 + 0*I;
        const float complex factor = 0 + 3*I;
        for (size_t r1 = 1; r1 < length-squaredmedian; r1++) {
                // From 1, since exp(i*log(0)) == 0
                integral += cexp(factor*log(r1))
                        * data[r1+squaredmedian];
        }
        return cabs(integral);
}

float TFunctional5(const float* data, const size_t length)
{
        // Transform the domain from t to r1
        size_t squaredmedian = findWeighedMedianSqrt(data, length);

        // Integrate
        float complex integral = 0 + 0*I;
        const float complex factor = 0 + 4*I;
        for (size_t r1 = 1; r1 < length-squaredmedian; r1++) {
                // From 1, since exp(i*log(0)) == 0
                integral += cexp(factor*log(r1))
                        * (sqrt(r1)*data[r1+squaredmedian]);
        }
        return cabs(integral);
}


//
// P-functionals
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

float PFunctional2(const float* data, const size_t length)
{
        size_t median = findWeighedMedian(data, length);
        return data[median];
}

float PFunctional3(const float* data, const size_t length)
{
        // Calculate the Fourier transform
        float complex *fourier = (float complex *) malloc(length * sizeof(float complex));
        for(size_t i = 0; i < length; i++) {
                fourier[i] = 0 + 0*I;
                float arg = -2.0 * M_PI * (float)i / (float)length;
                for(size_t j = 0; j < length; j++) {
                        float cosarg = cos(j * arg);
                        float sinarg = sin(j * arg);
                        fourier[i] += (float)data[j] * cosarg
                                + (float)data[j] * sinarg * I;
                }
        }

        // Integrate
        float sum = 0;
        for (size_t p = 0; p < length; p++)
                sum += pow(cabs(fourier[p]), 4);
        free(fourier);
        return sum;
}

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
