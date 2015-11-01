#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mwIndex *ir, *jc;
	double *values;
	int m = (int) mxGetM(prhs[0]);
	int n = (int) mxGetN(prhs[0]);
	values = mxGetPr(prhs[0]);
	ir = mxGetIr(prhs[0]);
	jc = mxGetJc(prhs[0]);
	long nnz = (long)mxGetNzmax(prhs[0]);

	int m1 = mxGetM(prhs[1]);
	int n1 = mxGetN(prhs[1]);
	double *uvalues = mxGetPr(prhs[1]);

	int m2 = mxGetM(prhs[2]);
	int n2 = mxGetN(prhs[2]);
	double *vvalues = mxGetPr(prhs[2]);
	int k = m1;

	plhs[0] = mxCreateSparse(m, n, nnz, mxREAL);
	mwIndex *start_of_ir = mxGetIr(plhs[0]);
	memcpy(start_of_ir, ir, nnz*sizeof(mwIndex));
	mwIndex *start_of_jc = mxGetJc(plhs[0]);
	memcpy(start_of_jc, jc, (n+1)*sizeof(mwIndex));
	double *res_values = mxGetPr(plhs[0]);

	for ( int j=0 ; j<n ; j++)
		for ( long idx = jc[j] ; idx < jc[j+1] ; idx++) 
		{
			int i = ir[idx];
			res_values[idx] = 0;
			for ( int t=0, ii = i*k, jj=j*k ; t<k ; t++, ii++, jj++ )
				res_values[idx] += (uvalues[ii]*vvalues[jj]);
		}
}
