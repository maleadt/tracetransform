//
// Configuration
//

// Include guard
#ifndef _TRACETRANSFORM_FUNCTIONALS_
#define _TRACETRANSFORM_FUNCTIONALS_

// Local
#include "global.hpp"

//
// Auxiliary
//

int findWeightedMedian(const Eigen::VectorXf& data);
int findWeightedMedianSquared(const Eigen::VectorXf& data);


//
// T functionals
//

// Radon
float TFunctionalRadon(const Eigen::VectorXf& data);

// T1
float TFunctional1(const Eigen::VectorXf& data);

// T2
float TFunctional2(const Eigen::VectorXf& data);

// T3, T4 and T5
typedef struct {
    float *real;
    float *imag;
} TFunctional345_precalc_t;
TFunctional345_precalc_t *TFunctional3_prepare(int rows, int cols);
TFunctional345_precalc_t *TFunctional4_prepare(int rows, int cols);
TFunctional345_precalc_t *TFunctional5_prepare(int rows, int cols);
float TFunctional345(const Eigen::VectorXf& data,
                     TFunctional345_precalc_t *precalc);
void TFunctional345_destroy(TFunctional345_precalc_t *precalc);

// T6
float TFunctional6(const Eigen::VectorXf& data);

// T7
float TFunctional7(const Eigen::VectorXf& data);


//
// P-functionals
//

// P1
float PFunctional1(const Eigen::VectorXf& data);

// P2
float PFunctional2(const Eigen::VectorXf& data);

// P3
typedef void PFunctional3_precalc_t;
PFunctional3_precalc_t *PFunctional3_prepare(int rows);
float PFunctional3(const Eigen::VectorXf& data);
void PFunctional3_destroy(PFunctional3_precalc_t *precalc);

// Hermite P-functionals
float PFunctionalHermite(const Eigen::VectorXf& data,
                         unsigned int order, int center);

#endif
