#include "mex.h" /* Always include this */
#include "matrix.h"
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                 int nrhs, const mxArray *prhs[]) /* Input variables */
{
    /*Variable declaration*/
    /*Dimensional variables*/
    mwSize M,N;   /*Variables of dimension of the input image*/ 
    
    /*mxArray variables*/
    mxArray *Input_data[3], *Rotated_image[2], *Input_Tfunct[2], *Input_Stacking[3],*Output_Stacking[1], *StackSinog[3], *Nonorm_CircusF[1];
    
    /*Regular variables*/
    double *angle, *T_codes, *P_codes,*stack_idx;/* Auxiliar pointers*/
    int k, n, c, dims[3];                        /* Loop indices*/
    double angle_intrvl;                         /* Angle of interval*/
    int Numb_rotations,Nang; 
    char Tfunct[13]="FunctionalT1";
    char Pfunct[8]="functP1";
    char base = 48;
    int Numb_Tfunct, Numb_Pfunct;               /*Number of T and P functionals;*/ 
    
     /*Parsing inputs*/
    if(nrhs != 5 || mxGetNumberOfDimensions(prhs[0]) != 2 || !mxIsDouble(prhs[0]))
    mexErrMsgTxt("Invalid input.");
    
    /*Dimensions of input data*/
    M = (mwSize) mxGetM(prhs[0]);           /*Number of rows of the input image*/
    N = (mwSize) mxGetN(prhs[0]);           /*Number of columns of the input image*/
    Numb_Tfunct = (int)mxGetN(prhs[1]);     /*Number of T-functionals*/
    Numb_Pfunct = (int)mxGetN(prhs[2]);     /*Number of P-functionals*/
    
    /*Additional Data Dimensions */
    angle_intrvl = mxGetScalar(prhs[3]);
    Numb_rotations = (int)(360.0/(angle_intrvl*4.0)); /*Number of rotations*/
    Nang = (int)(360.0/angle_intrvl);                 /* Number of angles*/
    dims[0] = (int)N; dims[1] = Nang; dims[2] = Numb_Tfunct;
    
    
    /*Initialization of the Array variables*/
    Input_data[0] = mxCreateDoubleMatrix(M, N, mxREAL);
    Input_data[1] = mxCreateDoubleScalar(0.0);
    Input_data[2] = mxCreateString("crop");
    Input_Stacking[1] = mxCreateDoubleMatrix(1, 4, mxREAL);
    Input_Stacking[2] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
    Input_Tfunct[0] = mxCreateDoubleMatrix(M, N, mxREAL);
    plhs[0] = mxCreateDoubleMatrix(Nang, Numb_Tfunct*Numb_Pfunct, mxREAL);
    
    
    /*Getting pointers*/
     angle = mxGetPr(Input_data[1]);
     T_codes = mxGetPr(prhs[1]);
     P_codes = mxGetPr(prhs[2]);
     stack_idx = mxGetPr(Input_Stacking[1]);
     
     /*Getting a copy of the input image*/
    memcpy(mxGetPr(Input_data[0]), mxGetPr(prhs[0]), sizeof(double)*M*N);
     
     
     
     for(k = 0; k < Numb_rotations; k++){       /*Loop to rotate the image*/
        
        *angle = (double)(angle_intrvl*k);      /* Angle in which the image will be rotated*/
     
        mexCallMATLAB(1, &Rotated_image[0], 3, Input_data, "imrotate");      /*Rotating the input image*/
        mexCallMATLAB(1, &Rotated_image[1], 1, &Rotated_image[0],"rotflip")     /*Rotated, flipped, and transponsed image */;
                
        for(n = 0; n < 2; n++){                 /*Loop to compute the vector shifts of the rotated and its transponsed*/
            
            mexCallMATLAB(1, &Input_Tfunct[1], 1, &Rotated_image[n], "vecWM");              /*Finding the the vector shift with weighted median*/                     
            memcpy(mxGetPr(Input_Tfunct[0]), mxGetPr(Rotated_image[n]), sizeof(double)*N*M);/*Storing the rotated image and the shift vector to apply T functional*/
            
            
            for(c = 0; c < Numb_Tfunct; c++){   /*Loop to apply the T-functional*/
                
                Tfunct[11] = (char)((int)base + (int)*(T_codes + c));
                mexCallMATLAB(1, &Input_Stacking[0], 2, Input_Tfunct, Tfunct);  /*Compute the T-functional*/
                *stack_idx = (double)k;*(stack_idx+1) = (double)n;*(stack_idx+2) = (double)c;*(stack_idx+3) = (double)Numb_rotations;  
                mexCallMATLAB(1,Output_Stacking,3,Input_Stacking,"Stacking");   /*Stores the T functional to compose the sinogram*/
                memcpy(mxGetPr(Input_Stacking[2]), mxGetPr(Output_Stacking[0]), sizeof(double)*N*Nang*Numb_Tfunct);
            }
        }
     }
    
     plhs[0] = Input_Stacking[2];
    
    /*Destroying initialized variables*/ 
    mxDestroyArray(Input_data[0]);
    mxDestroyArray(Input_data[1]);
    mxDestroyArray(Input_Stacking[1]);
    mxDestroyArray(Input_Tfunct[0]);
}