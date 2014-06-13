/*Functional T(f(x))=Int[0-inf] f(x)dx  */

#include "mex.h" /* Always include this */
#include "matrix.h"
#include <math.h>
#include <stdio.h>

void functional(double *pin, int M, double pout[]) {
    int k;
    pout[0] = 0;
    for (k = 0; k < M; k++) {
        pout[0] += *(pin + k);
    }
    pout[1] = pout[0];
}

void mexFunction(int nlhs, mxArray *plhs[],       /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
    mwSize M, N;
    double *pSign, *ptrI, ptr_out[2] = { 0.0, 0.0 };
    int k;

    /*Parsing inputs*/
    if (nrhs != 1 || mxGetNumberOfDimensions(prhs[0]) != 2 ||
        !mxIsDouble(prhs[0]))
        mexErrMsgTxt("Invalid input.");
    /* NOTE: although we get 2 inputs, can only use the first one */

    M = (mwSize)mxGetM(prhs[0]);
    N = (mwSize)mxGetN(prhs[0]);

    plhs[0] = mxCreateDoubleMatrix(N, 2, mxREAL);
    pSign = mxGetPr(plhs[0]);
    ptrI = mxGetPr(prhs[0]);
    for (k = 0; k < N; k++) {

        functional((ptrI + k * M), M, ptr_out);
        *(pSign + k) = ptr_out[0]; /*Dividing by 150 to get the an average*/
        *(pSign + k + N) = ptr_out[1];
    }
}
