#include "mex.h" /* Always include this */
#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void normalization(double *ptrD, double *ptrW, double *ptrZ0, double *ptrZ1, int M)
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
        *(ptrZ0 + j + 2*M) = *(ptrD + j);
        *(ptrZ0 + j + 3*M) = sqrt((*(ptrW + j)))/Sum[1]; 
        }
    
        *ptrZ1 = 1.0;
        *(ptrZ1+1) = 3.0;
}

void finding(double *ptrAsrt, double *ptrWMed, int M, int k)
{
    int i,j;
    double flag, cumsum[2]={0.0,0.0};
    for(i = 0; i < 2; i++){
        flag = 0.0;
        for(j = 0; j < M; j++){
            cumsum[i] += *(ptrAsrt+j+(2*i+1)*M);
            if (cumsum[i]>=0.5){
            *(ptrWMed+2*k+i) = *(ptrAsrt+j+2*i*M);
            flag = 1.0;
            break;}
        }
        if(flag == 0.0){
        *(ptrWMed+2*k+i) = 0.0;
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
    mwSize M, N;
    mxArray *Z[2];
    double *ptrD, *ptrW, *ptrZ0, *ptrZ1, *ptrWMed;
    int i;
    
    /*Parsing inputs*/
    if(nrhs != 2 || !mxIsDouble(prhs[0]))
    mexErrMsgTxt("Invalid input.");
    
    M = (mwSize) mxGetM(prhs[0]);
    N = (mwSize) mxGetN(prhs[0]);
    
    Z[0] = mxCreateDoubleMatrix(M, 4, mxREAL);
    Z[1] = mxCreateDoubleMatrix(1, 2, mxREAL);
    plhs[0] = mxCreateDoubleMatrix(2, N, mxREAL);
    ptrD = mxGetPr(prhs[0]);
    ptrW = mxGetPr(prhs[1]);
    
    for(i = 0; i < N; i++){
        ptrZ0 = mxGetPr(Z[0]);
        ptrZ1 = mxGetPr(Z[1]);
        normalization((ptrD+i*M), (ptrW+i*M), ptrZ0, ptrZ1, M);
        ptrZ0 = mxGetPr(Z[0]);
        ptrWMed = mxGetPr(plhs[0]);
        finding(ptrZ0, ptrWMed, M, i);;
    }
  
    mxDestroyArray(Z[0]);
    mxDestroyArray(Z[1]);
}
