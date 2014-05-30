/* median{rf(r),f(r)*/

#include "mex.h" /* Always include this */
#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include "vecWMfinal.c"

void functional(double *pin, double c, int M, double *pout) {
  int k, cint;
  mxArray *proj1[2], *proj2[2], *WM[2];
  double *dkernel1, *dkernel2, *wkernel1, *wkernel2, *pWM;
  cint = (int)c;
  proj1[0] = mxCreateDoubleMatrix(M - cint, 1, mxREAL);
  proj2[0] = mxCreateDoubleMatrix(cint, 1, mxREAL);
  proj1[1] = mxCreateDoubleMatrix(M - cint, 1, mxREAL);
  proj2[1] = mxCreateDoubleMatrix(cint, 1, mxREAL);
  dkernel1 = mxGetPr(proj1[0]);
  dkernel2 = mxGetPr(proj2[0]);
  wkernel1 = mxGetPr(proj1[1]);
  wkernel2 = mxGetPr(proj2[1]);

  for (k = cint; k < M; k++) {
    *(dkernel1 + k - cint) = *(pin + k);
    *(wkernel1 + k - cint) = sqrt(*(pin + k));
  }
  for (k = 0; k < cint; k++) {
    *(dkernel2 + k) = *(pin + cint - k - 1);
    *(wkernel2 + k) = sqrt(*(pin + cint - k - 1));
  }
  vecWMfinal(1, &WM[0], 2, (const mxArray **)proj1);
  vecWMfinal(1, &WM[1], 2, (const mxArray **)proj2);
  pWM = mxGetPr(WM[0]);
  pout[0] = *pWM;
  pWM = mxGetPr(WM[1]);
  pout[1] = *pWM;
  mxDestroyArray(proj1[0]);
  mxDestroyArray(proj1[1]);
  mxDestroyArray(proj2[0]);
  mxDestroyArray(proj2[1]);
}

void mexFunction(int nlhs, mxArray *plhs[],       /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
  mwSize M, N;
  double *pSign, *ptrI, *ptr_c, ptr_out[2];
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
    *(pSign + k) = ptr_out[0];
    *(pSign + k + N) = ptr_out[1];
  }
}
