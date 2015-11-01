/* element-wise multiply of two double arrays of exactly the same size, and then sums along the columns for observed entries*/

/* creates P_\omega(UV') , where U(nrows \times rank), V(ncols \times rank), omega=[irow,jcol] 
outputs a vector "proj" such that [irow,jcol,proj] is the projected  matrix
>>> calling sequnce is : project_obs_UV(U,V,irow,jcol,no_obs)
This is a .c file, to handle this Projection operation more efficiently than MATLAB's array accessing, for large problems 
*/
 
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{
   int i,j, k, jj, no_obs, rank; 
   double *U, *V, *proj, *irow, *jcol, tempv; 
   
   mwSize nrows, ncols; 

   U = mxGetPr(prhs[0]);
   V = mxGetPr(prhs[1]);
   irow= mxGetPr(prhs[2]);
   jcol= mxGetPr(prhs[3]);
   no_obs= mxGetScalar(prhs[4]);

/* Get dimensions, length of input vector */
   nrows=mxGetM(prhs[0]);
   rank=mxGetN(prhs[0]);
   ncols=mxGetM(prhs[1]);
 /*  no_obs=mxGetN(prhs[3]); */

/* Allocate memory and assign output pointer */
plhs[0] = mxCreateDoubleMatrix(no_obs, 1, mxREAL); 
/*mxReal is our data-type*/

 /*  create a C pointer to a copy of the output matrix */
  proj = mxGetPr(plhs[0]);


/* this  evaluates the sum(.*) */
   for (k=0; k<no_obs; k++ )  {

i=irow[k]-1;
j=jcol[k]-1;
tempv=0;
    for (jj=0; jj< rank; jj++) {
       tempv += U[i + (jj*nrows)] * V[j + (jj*ncols)];
                                } 
proj[k]=tempv;       

                           }


}

