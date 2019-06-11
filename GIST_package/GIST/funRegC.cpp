#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mex.h>
#include <math.h>
#include <algorithm>
#include "matrix.h"

/*
-------------------------- Function funRegC -----------------------------

 f = \sum_i r_i(x)

 Usage (in matlab):
 [f]=funRegC(x, n, lambda, theta,type); n is the dimension of the vector d

 type = 1: Capped L1 regularizer (CapL1) 
           r_i(x) = lambda*\min(|x_i|,theta), (theta > 0, lambda >= 0)

 type = 2: Log Sum Penalty (LSP)
           r_i(x) = lambda*\sum_i log(1 + |x_i|/theta), (theta > 0, lambda >= 0)

 type = 3: TNN


 default: type = 1

-------------------------- Function funRegC -----------------------------

-------------------------- Reference -----------------------------------------

[1] Pinghua Gong, Changshui Zhang, Zhaosong Lu, Jianhua Huang, Jieping Ye,
    A General Iterative Shrinkage and Thresholding Algorithm for Non-convex
    Regularized Optimization Problems. ICML 2013.

-----------------------------------------------------------------------------

   Copyright (C) 2012-2013 Pinghua Gong

   For any problem, please contact Pinghua Gong via pinghuag@gmail.com
*/


void funCapL1(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	double u = 0.0;;
	for(i=0;i<n;i++)
	{   
		u += std::min(fabs(x[i]),theta);
	}
	*f = u*lambda;
	return;
}

void funLSP(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	double u = 0.0;

	for(i=0;i<n;i++)
	{ 
		u += log(1.0 + fabs(x[i])/theta);
	}
	*f = u*lambda;
	return;
}

void funTNN(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	double u = 0.0;

	for(i = theta - 1; i < n; i++)
	{ 
		u += x[i];
	}
	*f = u*lambda;
	return;
}

/*void funSCAD(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	double u,v,y,z,w;
	y = theta*lambda;
	w = lambda*lambda;
	z = 0.5*(theta+1.0)*w;

	u = 0.0;
	for(i=0;i<n;i++)
	{ 
		v = fabs(x[i]);
		if (v <= lambda) 
			u += lambda*v;
		else if (v > y)
			u += z;
		else
			u += 0.5*(v*(2*y - v) - w)/(theta-1.0);
	}	
	
	*f = u;
	return;
}*/

/*void funMCP(double *f, double *x, long n, double lambda, double theta)
{
    long i;
	double v,u,y;
	y = theta*lambda;
	u = 0.0;
    for(i=0;i<n;i++)
    { 
		v = fabs(x[i]);
		if (v <= y)
			u += v*(lambda - 0.5*v/theta);
		else
			u += 0.5*y*lambda;
	}
	*f = u;
	return;
}*/




void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /*set up input arguments */
    double* x       =            mxGetPr(prhs[0]);
    long     n      =      (long)mxGetScalar(prhs[1]);
    double  lambda  =            mxGetScalar(prhs[2]);
	double  theta   =            mxGetScalar(prhs[3]);
	int     type    =       (int)mxGetScalar(prhs[4]);
    
    /* set up output arguments */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    double *f = mxGetPr(plhs[0]);

	switch (type)
	{
	case 1:
		funCapL1(f, x, n, lambda, theta);
		break;
	case 2:
		funLSP(f, x, n, lambda, theta);
		break;
	case 3:
		funTNN(f, x, n, lambda, theta);
		break;
	default:
		mexPrintf("not support! \n");
	}
}


