#include "mex.h" /* Always include this */
#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void normalization(double *ptrD, double *ptrW, double *ptrZ0, int M) {
    int j;
    double Sum[2] = { 0.0, 0.0 };
    for (j = 0; j < M; j++) {
        Sum[0] += *(ptrW + j);
        Sum[1] += sqrt(*(ptrW + j));
    }

    for (j = 0; j < M; j++) {
        *(ptrZ0 + j) = *(ptrD + j);
        *(ptrZ0 + j + M) = (*(ptrW + j)) / Sum[0];
    }
}

void finding(double *ptrAsrt, double *ptrWMed, int M, int k) {
    int j;
    double flag, cumsum = 0.0;
    flag = 0.0;
    for (j = 0; j < M; j++) {
        cumsum += *(ptrAsrt + j + M);
        if (cumsum >= 0.5) {
            *(ptrWMed + k) = *(ptrAsrt + j);
            flag = 1.0;
            break;
        }
    }
    if (flag == 0.0) {
        *(ptrWMed + k) = 0.0;
    }
}

int compareIndexed(const void *a, const void *b, void *arg) {
    size_t *x = (size_t *)a;
    size_t *y = (size_t *)b;
    double *data = (double *)arg;

    double *u = data + *x;
    double *v = data + *y;

    if (*u < *v) {
        return -1;
    } else if (*u > *v) {
        return 1;
    } else {
        return 0;
    }
}

void sortrows(double *ptrZ0, size_t M, double *ptrAsrt) {
    size_t data_index[M], i, j;
    double data[M];
    for (i = 0; i < M; i++) {
        data_index[i] = i;
        data[i] = *(ptrZ0 + i);
    }
    qsort_r(data_index, M, sizeof(size_t), compareIndexed, &data);
    for (j = 0; j < M; j++) {
        *(ptrAsrt + j) = *(ptrZ0 + data_index[j]);
        *(ptrAsrt + M + j) = *(ptrZ0 + M + data_index[j]);
    }
}

void
#if __INCLUDE_LEVEL__ == 0
mexFunction
#else
vecWMfinal
#endif
    (int nlhs, mxArray *plhs[],       /* Output variables */
     int nrhs, const mxArray *prhs[]) /* Input variables */
{
    mwSize M, N;
    mxArray *Z[1], *Asrt[1];
    double *ptrD, *ptrW, *ptrZ0, *ptrWMed, *ptrAsrt;
    int i, j;

    /*Parsing inputs*/
    if (nrhs != 2 || !mxIsDouble(prhs[0]))
        mexErrMsgTxt("Invalid input.");

    M = (mwSize)mxGetM(prhs[0]);
    N = (mwSize)mxGetN(prhs[0]);

    Z[0] = mxCreateDoubleMatrix(M, 2, mxREAL);
    Asrt[0] = mxCreateDoubleMatrix(M, 2, mxREAL);

    plhs[0] = mxCreateDoubleMatrix(1, N, mxREAL);
    ptrD = mxGetPr(prhs[0]);
    ptrW = mxGetPr(prhs[1]);

    for (i = 0; i < N; i++) {
        ptrZ0 = mxGetPr(Z[0]);
        ptrAsrt = mxGetPr(Asrt[0]);
        normalization((ptrD + i * M), (ptrW + i * M), ptrZ0, M);
        sortrows(ptrZ0, M, ptrAsrt);
        ptrWMed = mxGetPr(plhs[0]);
        finding(ptrAsrt, ptrWMed, M, i);
    }

    mxDestroyArray(Z[0]);
    mxDestroyArray(Asrt[0]);
}
