/*Functional T(f(x))=Int[0-inf] r*f(r)dr  */

#include "mex.h" /* Always include this */
#include "matrix.h"
#include <math.h>
#include <stdio.h>

void functional(double *pin, double c, int M, double pout[]) {
  int k, cint;
  cint = (int)c;
  pout[0] = 0;
  pout[1] = 0;
  for (k = cint; k < M; k++) {
    pout[0] += (k - cint) * (*(pin + k));
  }
  for (k = 0; k < cint; k++) {
    pout[1] += k * (*(pin + cint - k - 1));
  }
}

void mexFunction(int nlhs, mxArray *plhs[],       /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
  mwSize M, N;
  double *pSign, *ptrI, *ptr_c, ptr_out[2] = { 0.0, 0.0 };
  int k;

  /*Parsing inputs*/
  if (nrhs != 2 || mxGetNumberOfDimensions(prhs[0]) != 2 ||
      !mxIsDouble(prhs[0]))
    mexErrMsgTxt("Invalid input.");

  M = (mwSize)mxGetM(prhs[0]);
  N = (mwSize)mxGetN(prhs[0]);
  ptr_c = mxGetPr(prhs[1]);

  plhs[0] = mxCreateDoubleMatrix(N, 2, mxREAL);
  pSign = mxGetPr(plhs[0]);
  ptrI = mxGetPr(prhs[0]);
  for (k = 0; k < N; k++) {

    functional((ptrI + k * M), *(ptr_c + 2 * k), M, ptr_out);
    *(pSign + k) = ptr_out[0]; /*Dividing by 150 to get the an average*/
    *(pSign + k + N) = ptr_out[1];
  }
}
