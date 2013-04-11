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

size_t findWeighedMedian(const double* data, const size_t length)
{               
        double sum = 0;
        for (size_t i = 0; i < length; i++)
                sum += data[i];

        double integral = 0;
        for (size_t i = 0; i < length; i++) {
                integral += data[i];
                if (2*integral >= sum)
                        return i;
        }
        return length-1;
}

size_t findWeighedMedianSqrt(const double* data, const size_t length)
{               
        double sum = 0;
        for (size_t i = 0; i < length; i++)
                sum += sqrt(data[i]);

        double integral = 0;
        for (size_t i = 0; i < length; i++) {
                integral += sqrt(data[i]);
                if (2*integral >= sum)
                        return i;
        }

        return length-1;
}

double hermite_polynomial(unsigned int order, double x) {
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

double hermite_function(unsigned int order, double x) {
        return hermite_polynomial(order, x) / (
                        sqrt(pow(2, order) * factorial(order) * sqrt(M_PI))
                        * exp(x*x/2)
                );
}


//
// T functionals
//

// T-functional for the Radon transform.
double TFunctionalRadon(const double* data, const size_t length)
{
        double integral = 0;
        for (size_t t = 0; t < length; t++)
                integral += data[t];
        return integral;                
}

double TFunctional1(const double* data, const size_t length)
{
        // Transform the domain from t to r
        size_t median = findWeighedMedian(data, length);

        // Integrate
        double integral = 0;
        for (size_t r = 0; r < length-median; r++)
                integral += data[r+median] * r;
        return integral;                
}

double TFunctional2(const double* data, const size_t length)
{
        // Transform the domain from t to r
        size_t median = findWeighedMedian(data, length);

        // Integrate
        double integral = 0;
        for (size_t r = 0; r < length-median; r++)
                integral += data[r+median] * r*r;
        return integral;                
}

double TFunctional3(const double* data, const size_t length)
{
        // Transform the domain from t to r1
        size_t squaredmedian = findWeighedMedianSqrt(data, length);

        // Integrate
        double complex integral = 0 + 0*I;
        const double complex factor = 0 + 5*I;
        for (size_t r1 = 1; r1 < length-squaredmedian; r1++) {
                // From 1, since exp(i*log(0)) == 0
                integral += cexp(factor*log(r1))
                        * (r1*data[r1+squaredmedian]);

        }
        return cabs(integral);
}

double TFunctional4(const double* data, const size_t length)
{
        // Transform the domain from t to r1
        size_t squaredmedian = findWeighedMedianSqrt(data, length);

        // Integrate
        double complex integral = 0 + 0*I;
        const double complex factor = 0 + 3*I;
        for (size_t r1 = 1; r1 < length-squaredmedian; r1++) {
                // From 1, since exp(i*log(0)) == 0
                integral += cexp(factor*log(r1))
                        * data[r1+squaredmedian];
        }
        return cabs(integral);
}

double TFunctional5(const double* data, const size_t length)
{
        // Transform the domain from t to r1
        size_t squaredmedian = findWeighedMedianSqrt(data, length);

        // Integrate
        double complex integral = 0 + 0*I;
        const double complex factor = 0 + 4*I;
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

double PFunctional1(const double* data, const size_t length)
{
        double sum = 0;
        double previous = data[0];
        for (size_t p = 1; p < length; p++) {
                double current = data[p];
                sum += fabs(previous - current);
                previous = current;
        }
        return sum;
}

double PFunctional2(const double* data, const size_t length)
{
        size_t median = findWeighedMedian(data, length);
        return data[median];
}

double PFunctional3(const double* data, const size_t length)
{
        // Calculate the Fourier transform
        double complex *fourier = (double complex *) malloc(length * sizeof(double complex));
        for(size_t i = 0; i < length; i++) {
                fourier[i] = 0 + 0*I;
                double arg = -2.0 * M_PI * (double)i / (double)length;
                for(size_t j = 0; j < length; j++) {
                        double cosarg = cos(j * arg);
                        double sinarg = sin(j * arg);
                        fourier[i] += (double)data[j] * cosarg
                                + (double)data[j] * sinarg * I;
                }
        }

        // Integrate
        double sum = 0;
        for (size_t p = 0; p < length; p++)
                sum += pow(cabs(fourier[p]), 4);
        free(fourier);
        return sum;
}

double PFunctionalHermite(const double* data, const size_t length, unsigned int order, size_t center)
{
        // Discretize the [-10, 10] domain to fit the column iterator
        double z = -10;
        double stepsize_lower = 10.0 / center;
        double stepsize_upper = 10.0 / (length - 1 - center);

        // Calculate the integral
        double integral = 0;
        for (size_t p = 0; p < length; p++) {
                integral += data[p] * hermite_function(order, z);
                if (z < 0)
                        z += stepsize_lower;
                else
                        z += stepsize_upper;
        }
        return integral;
}
