#include "mex.h" /* Always include this */
#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


void normalization(double *ptrD, double *ptrW, double *ptrZ0, int M)
{
    int j;
    double Sum[2] = {0.0,0.0};
        for(j = 0; j < M; j++){
            Sum[0] += *(ptrW + j); 
            Sum[1] += sqrt(*(ptrW + j));
        }

        for(j = 0; j < M; j++){
        *(ptrZ0 + j) = *(ptrD + j);
        *(ptrZ0 + j + M) = (*(ptrW + j))/Sum[0]; 
        }
}

void finding(double *ptrAsrt, double *ptrWMed, int M, int k)
{
    int j;
    double flag, cumsum=0.0;
        flag = 0.0;
        for(j = 0; j < M; j++){
            cumsum += *(ptrAsrt+j+M);
            if (cumsum>=0.5){
            *(ptrWMed+k) = *(ptrAsrt+j);
            flag = 1.0;
            break;}
        }
        if(flag == 0.0){
        *(ptrWMed+k) = 0.0;
        }
    
}

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
    mwSize M, N;
    mxArray *Z[1], *Asrt[1];
    double *ptrD, *ptrW, *ptrZ0, *ptrWMed, *ptrAsrt;
    int i,j;
    
    /*Parsing inputs*/
    if(nrhs != 2 || !mxIsDouble(prhs[0]))
    mexErrMsgTxt("Invalid input.");
    
    M = (mwSize) mxGetM(prhs[0]);
    N = (mwSize) mxGetN(prhs[0]);
    
    Z[0] = mxCreateDoubleMatrix(M, 2, mxREAL);
    Asrt[0] = mxCreateDoubleMatrix(M, 2, mxREAL);
    
    plhs[0] = mxCreateDoubleMatrix(1, N, mxREAL);
    ptrD = mxGetPr(prhs[0]);
    ptrW = mxGetPr(prhs[1]);
    
    for(i = 0; i < N; i++){
        ptrZ0 = mxGetPr(Z[0]);
        normalization((ptrD+i*M), (ptrW+i*M), ptrZ0, M);
        mexCallMATLAB(1, Asrt, 1, Z, "sortrows");
        ptrAsrt = mxGetPr(Asrt[0]);
        ptrWMed = mxGetPr(plhs[0]);
        finding(ptrAsrt, ptrWMed, M, i);
    }
  
    mxDestroyArray(Z[0]);
    mxDestroyArray(Asrt[0]);
}
