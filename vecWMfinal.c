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

int sort(const void *x, const void *y) {
  return (*(int*)x - *(int*)y);
}

void vecsort(double *ptrZ0, double *ptrAsrt, int M)
{   
    int k,n;
    float *ptrtmp,cm,ch;
    mxArray *tmp[1];
    tmp[0] =  mxCreateNumericMatrix(M, 1, mxSINGLE_CLASS, mxREAL);
    ptrtmp = (float*)mxGetPr(tmp[0]);
        for(k = 0; k < M; k++){
            *(ptrtmp+k) = (float)*(ptrZ0+k);
        }
    qsort(ptrtmp, M, sizeof(float), sort);
    for (k = 0; k < M; k++){
        n = 0;
        cm = (float)*(ptrZ0+n+M);
        ch = *(ptrtmp+k);
        while (cm != ch){
        n= n+1;
        cm = (float)*(ptrZ0+n+M);
        printf("%d \n",n);
        }
        
        *(ptrAsrt+k) = (double)*(ptrtmp+k);
        printf("%f \n",*(ptrAsrt+k));
        *(ptrAsrt+ k + M) = (double)*(ptrZ0+n+M);
        printf("%f \n",*(ptrZ0+k+M));
    }
    mxDestroyArray(tmp[0]);
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
        /*mexCallMATLAB(1, Asrt, 1, Z, "sortrows");
        mexCallMATLAB(1, Asrt, 1, Z, "srtLst");*/
        
        ptrAsrt = mxGetPr(Asrt[0]);
        ptrWMed = mxGetPr(plhs[0]);
        
        /*vecsort(ptrZ0,ptrAsrt,M);
        for(j = 0; j < 2*M; j++){
             printf("%f \n",*(ptrAsrt+j));}
        finding(ptrAsrt, ptrWMed, M, i);*/
        finding(ptrZ0, ptrWMed, M, i);
    }
  
    mxDestroyArray(Z[0]);
    mxDestroyArray(Asrt[0]);
}
