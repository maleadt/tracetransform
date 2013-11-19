/*Functional T(f(x))=Int[0-inf] (exp(i*k*log(r)))*r^p*f(r)dr   p=0,k=3  */

#include "mex.h" /* Always include this */
#include "matrix.h"
#include <math.h>
#include <stdio.h>

void functional(double *pin, double c, int M, double *kernReal, double *kernIm,
                double *pout) {
  int k, cint;
  double p0Real, p0Im, p1Real, p1Im;
  cint = (int)c;
  p0Real = 0.0;
  p0Im = 0.0;
  p1Real = 0.0;
  p1Im = 0.0;
  for (k = cint; k < M; k++) {
    p0Real += *(kernReal + k - cint) * (*(pin + k));
    p0Im += *(kernIm + k - cint) * (*(pin + k));
  }
  for (k = 0; k < cint; k++) {
    p1Real += *(kernReal + k) * (*(pin + cint - k - 1));
    p1Im += *(kernIm + k) * (*(pin + cint - k - 1));
  }
  pout[0] = sqrt(p0Real * p0Real + p0Im * p0Im);
  pout[1] = sqrt(p1Real * p1Real + p1Im * p1Im);
}

void mexFunction(int nlhs, mxArray *plhs[],       /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
  mwSize M, N;
  mxArray *kernel[1];
  double *pSign, *ptrI, *ptr_c, ptr_out[2], r, *ptrKreal, *ptrKim;
  int k;

  /*Parsing inputs*/
  if (nrhs != 2 || mxGetNumberOfDimensions(prhs[0]) != 2 ||
      !mxIsDouble(prhs[0]))
    mexErrMsgTxt("Invalid input.");

  M = (mwSize)mxGetM(prhs[0]);
  N = (mwSize)mxGetN(prhs[0]);
  ptr_c = mxGetPr(prhs[1]);

  plhs[0] = mxCreateDoubleMatrix(N, 2, mxREAL);
  kernel[0] = mxCreateDoubleMatrix(M, 1, mxCOMPLEX);
  pSign = mxGetPr(plhs[0]);
  ptrI = mxGetPr(prhs[0]);
  ptrKreal = mxGetPr(kernel[0]);
  ptrKim = mxGetPi(kernel[0]);

  for (k = 1; k <= N; k++) {
    r = (double)k;
    *(ptrKreal + k - 1) = cos(3.0 * log(r));
    *(ptrKim + k - 1) = sin(3.0 * log(r));
  }

  for (k = 0; k < N; k++) {

    functional((ptrI + k * M), *(ptr_c + 2 * k + 1), M, ptrKreal, ptrKim,
               ptr_out);
    *(pSign + k) = ptr_out[0];
    *(pSign + k + N) = ptr_out[1];
  }
  mxDestroyArray(kernel[0]);
}
